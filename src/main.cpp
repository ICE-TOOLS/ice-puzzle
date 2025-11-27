#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
#include <string>
#include <array>
#include <random>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <gmp.h>
#include <secp256k1.h>
#ifdef _WIN32
#include <windows.h>
#endif
#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

// --- Forward Declarations ---
bool base58check_decode(const std::string& s, uint8_t& version, std::array<uint8_t,20>& out_hash);
std::string base58check_encode(uint8_t version, const uint8_t* payload, size_t len);
void hash160(const uint8_t* data, size_t len, std::array<uint8_t,20>& out);
bool privkey_to_compressed_pubkey(const uint8_t privkey[32], std::array<uint8_t,33>& out_pubkey, secp256k1_context* ctx);
void mpz_to_32be(mpz_t v, uint8_t out[32]);
std::string mpz_to_hex(const mpz_t v);
std::string bytes_to_hex(const uint8_t* data, size_t len);
int be32_cmp(const uint8_t* a, const uint8_t* b);
void be32_inc(uint8_t* a);
std::string shorten_mid(const std::string& s, size_t keep_front, size_t keep_back);

// Default to no CUDA if not defined by build system
#ifndef HAVE_CUDA
#define HAVE_CUDA 0
#endif

// --- GPU FUNCTIONS (Must be extern "C") ---
extern "C" void gpu_hash160_batch(const uint8_t* pubkeys33, uint8_t* out_hashes20, size_t n);
extern "C" void gpu_pubkey_hash_from_priv_batch(const uint8_t* base_priv, uint8_t* out_pub33, uint8_t* out_hash20, size_t n);
#if HAVE_CUDA
extern "C" void gpu_pubkey_hash_from_priv_batch_stream(const uint8_t* base_priv, uint8_t* out_pub33, uint8_t* out_hash20, size_t n, cudaStream_t stream);
#endif

struct Segment { mpz_t s; mpz_t e; };

int main() {
  const std::string targetAddress = "1K3x5L6G57Y494fDqBfrojD28UJv4s5JcK";
  const char* startHex = "2000000000000000000000";
  const char* endHex = "3fffffffffffffffffffff";

  uint8_t version = 0;
  std::array<uint8_t,20> targetHash{};
  if (!base58check_decode(targetAddress, version, targetHash) || version != 0) {
    std::cerr << "Invalid target address" << std::endl;
    return 1;
  }

  mpz_t start; mpz_init(start); mpz_set_str(start, startHex, 16);
  mpz_t end; mpz_init(end); mpz_set_str(end, endHex, 16);

  if (mpz_cmp(start, end) > 0) {
    std::cerr << "Invalid range" << std::endl;
    mpz_clear(start); mpz_clear(end);
    return 1;
  }

  mpz_t total; mpz_init(total); mpz_sub(total, end, start); mpz_add_ui(total, total, 1);

  unsigned int tcount = std::thread::hardware_concurrency();
  if (tcount == 0) tcount = 1;

  mpz_t chunk; mpz_init(chunk);
  mpz_t rem; mpz_init(rem);
  mpz_fdiv_qr_ui(chunk, rem, total, tcount);
  unsigned long rem_ui = mpz_get_ui(rem);

  std::vector<Segment> segs(tcount);
  mpz_t cursor; mpz_init(cursor); mpz_set(cursor, start);
  for (unsigned int i = 0; i < tcount; ++i) {
    mpz_init(segs[i].s);
    mpz_init(segs[i].e);
    mpz_set(segs[i].s, cursor);
    mpz_t len; mpz_init(len);
    mpz_set(len, chunk);
    if (i < rem_ui) mpz_add_ui(len, len, 1);
    mpz_add(segs[i].e, segs[i].s, len);
    mpz_sub_ui(segs[i].e, segs[i].e, 1);
    mpz_add_ui(cursor, segs[i].e, 1);
    mpz_clear(len);
  }
  mpz_clear(cursor);

  std::vector<std::array<uint8_t,32>> segStartBytes(tcount);
  std::vector<std::array<uint8_t,32>> segEndBytes(tcount);
  for (unsigned int i = 0; i < tcount; ++i) {
    mpz_to_32be(segs[i].s, segStartBytes[i].data());
    mpz_to_32be(segs[i].e, segEndBytes[i].data());
  }

  std::atomic<bool> found{false};
  std::atomic<uint64_t> processed{0};
  std::string winnerPrivHex;
  std::string winnerAddr;

  std::atomic<bool> done{false};
  std::thread reporter([&](){
    using namespace std::chrono;
    uint64_t last = processed.load();
    auto t0 = steady_clock::now();
    while (!done.load()) {
      std::this_thread::sleep_for(seconds(1));
      uint64_t now = processed.load();
      auto t1 = steady_clock::now();
      double secs = duration_cast<duration<double>>(t1 - t0).count();
      double speed = (now - last) / secs;
      t0 = t1;
      last = now;
      std::cout << "\x1b]0;Speed: " << (uint64_t)speed << " keys/sec\x07";
#ifdef _WIN32
      {
        std::string title = "Speed: " + std::to_string((uint64_t)speed) + " keys/sec";
        SetConsoleTitleA(title.c_str());
      }
#endif
    }
  });

  std::vector<std::thread> workers;
  workers.reserve(tcount);
  std::mutex outMutex;
  for (unsigned int i = 0; i < tcount; ++i) {
    workers.emplace_back([&, i]() {
      // CPU context only used for verification if found
      secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN); 
      std::array<uint8_t,32> priv{};
      std::memcpy(priv.data(), segStartBytes[i].data(), 32);
      std::array<uint8_t,32> endb{};
      std::memcpy(endb.data(), segEndBytes[i].data(), 32);

      uint64_t local = 0;
      while (!found.load() && be32_cmp(priv.data(), endb.data()) <= 0) {
#if HAVE_CUDA
        const size_t batch = 65536;
        uint8_t* comps = nullptr;
        uint8_t* hashes = nullptr;
        cudaHostAlloc((void**)&comps, batch * 33, cudaHostAllocDefault);
        cudaHostAlloc((void**)&hashes, batch * 20, cudaHostAllocDefault);
        cudaStream_t stream; cudaStreamCreate(&stream);
        size_t remaining = batch;
        uint8_t basePriv[32]; std::memcpy(basePriv, priv.data(), 32);
        gpu_pubkey_hash_from_priv_batch_stream(basePriv, comps, hashes, remaining, stream);
        cudaStreamSynchronize(stream);
        for (size_t j = 0; j < remaining; ++j) {
          const uint8_t* h = &hashes[j*20];
          if (std::memcmp(h, targetHash.data(), 20) == 0) {
            uint8_t tmpPriv[32]; std::memcpy(tmpPriv, basePriv, 32);
            for (size_t s = 0; s < j; ++s) be32_inc(tmpPriv);
            std::string phex = bytes_to_hex(tmpPriv, 32);
            std::string fullAddr = base58check_encode(0x00, h, 20);
            if (!found.exchange(true)) {
              {
                std::lock_guard<std::mutex> lk(outMutex);
                winnerPrivHex = phex;
                winnerAddr = fullAddr;
                std::ofstream f("winner.txt", std::ios::out | std::ios::trunc);
                if (f) { f << "PrivateKeyHex=" << winnerPrivHex << "\n"; f << "Address=" << winnerAddr << "\n"; }
              }
              std::cout << "\x1b[92mFOUND " << winnerPrivHex << " " << winnerAddr << "\x1b[0m" << std::endl;
            }
            break;
          }
        }
        for(size_t s=0;s<remaining;s++) be32_inc(priv.data());
        cudaStreamDestroy(stream);
        cudaFreeHost(comps);
        cudaFreeHost(hashes);
        local += remaining;
#else
        secp256k1_pubkey pubkey;
        if (!secp256k1_ec_seckey_verify(ctx, priv.data())) break;
        if (!secp256k1_ec_pubkey_create(ctx, &pubkey, priv.data())) break;
        std::array<uint8_t,33> comp{}; size_t clen = 33;
        if (!secp256k1_ec_pubkey_serialize(ctx, comp.data(), &clen, &pubkey, SECP256K1_EC_COMPRESSED)) break;
        std::array<uint8_t,20> h{}; hash160(comp.data(), comp.size(), h);
        if (std::memcmp(h.data(), targetHash.data(), 20) == 0) {
          std::string phex = bytes_to_hex(priv.data(), 32);
          std::string fullAddr = base58check_encode(0x00, h.data(), h.size());
          if (!found.exchange(true)) {
            {
              std::lock_guard<std::mutex> lk(outMutex);
              winnerPrivHex = phex; winnerAddr = fullAddr;
              std::ofstream f("winner.txt", std::ios::out | std::ios::trunc);
              if (f) { f << "PrivateKeyHex=" << winnerPrivHex << "\n"; f << "Address=" << winnerAddr << "\n"; }
            }
            std::cout << "\x1b[92mFOUND " << winnerPrivHex << " " << winnerAddr << "\x1b[0m" << std::endl;
          }
          break;
        }
        {
          std::string phex = bytes_to_hex(priv.data(), 32);
          std::string pubhex = shorten_mid(bytes_to_hex(comp.data(), comp.size()), 10, 10);
          std::string hashhex = shorten_mid(bytes_to_hex(h.data(), h.size()), 8, 8);
          std::string addr = shorten_mid(base58check_encode(0x00, h.data(), h.size()), 6, 6);
          std::lock_guard<std::mutex> lk(outMutex);
          std::cout << "\x1b[36mKEY " << phex << " PUB " << pubhex << " HASH160 " << hashhex << " ADDR " << addr << "\x1b[0m" << std::endl;
        }
        be32_inc(priv.data());
        local++;
#endif
        if (local >= 10000) { processed.fetch_add(local); local = 0; }
      }
      if (local) processed.fetch_add(local);
      secp256k1_context_destroy(ctx);
    });
  }

  for (auto& th : workers) th.join();
  done.store(true);
  reporter.join();

  for (auto& seg : segs) { mpz_clear(seg.s); mpz_clear(seg.e); }
  mpz_clear(start); mpz_clear(end); mpz_clear(total); mpz_clear(chunk); mpz_clear(rem);

  if (!winnerPrivHex.empty()) return 0;
  std::cout << "Not found in range" << std::endl;
  return 0;
}
