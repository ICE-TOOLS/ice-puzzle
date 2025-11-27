#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <secp256k1.h>
#include <gmp.h>
#include <array>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>

static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static void sha256_once(const uint8_t* data, size_t len, uint8_t out[32]) {
  SHA256(data, len, out);
}

bool base58check_decode(const std::string& s, uint8_t& version, std::array<uint8_t,20>& out_hash) {
  std::array<int,256> map{};
  map.fill(-1);
  for (int i = 0; BASE58_ALPHABET[i] != 0; ++i) map[(uint8_t)BASE58_ALPHABET[i]] = i;
  size_t zeros = 0;
  while (zeros < s.size() && s[zeros] == '1') ++zeros;
  std::vector<uint8_t> b256;
  for (char c : s) {
    int val = map[(uint8_t)c];
    if (val < 0) return false;
    int carry = val;
    for (int i = (int)b256.size() - 1; i >= 0; --i) {
      int x = (int)b256[i] * 58 + carry;
      b256[i] = (uint8_t)(x & 0xFF);
      carry = x >> 8;
    }
    while (carry > 0) {
      b256.insert(b256.begin(), (uint8_t)(carry & 0xFF));
      carry >>= 8;
    }
  }
  std::vector<uint8_t> full;
  full.resize(zeros + b256.size());
  std::fill(full.begin(), full.begin() + zeros, 0);
  std::copy(b256.begin(), b256.end(), full.begin() + zeros);
  if (full.size() < 25) return false;
  uint8_t check1[32];
  sha256_once(full.data(), full.size() - 4, check1);
  uint8_t check2[32];
  sha256_once(check1, 32, check2);
  if (!std::equal(check2, check2 + 4, full.data() + full.size() - 4)) return false;
  version = full[0];
  for (int i = 0; i < 20; ++i) out_hash[i] = full[1 + i];
  return true;
}

std::string base58check_encode(uint8_t version, const uint8_t* payload, size_t len) {
  std::vector<uint8_t> data;
  data.reserve(1 + len + 4);
  data.push_back(version);
  for (size_t i = 0; i < len; ++i) data.push_back(payload[i]);
  uint8_t h1[32];
  sha256_once(data.data(), data.size(), h1);
  uint8_t h2[32];
  sha256_once(h1, 32, h2);
  for (int i = 0; i < 4; ++i) data.push_back(h2[i]);
  size_t zeros = 0;
  while (zeros < data.size() && data[zeros] == 0) ++zeros;
  std::vector<uint8_t> temp = data;
  std::string result;
  std::vector<uint8_t> b58;
  while (!temp.empty()) {
    int carry = 0;
    std::vector<uint8_t> q;
    q.reserve(temp.size());
    for (size_t i = 0; i < temp.size(); ++i) {
      int x = (carry << 8) | temp[i];
      int div = x / 58;
      carry = x % 58;
      if (!q.empty() || div != 0) q.push_back((uint8_t)div);
    }
    b58.push_back((uint8_t)carry);
    temp = q;
  }
  for (size_t i = 0; i < zeros; ++i) result.push_back('1');
  for (int i = (int)b58.size() - 1; i >= 0; --i) result.push_back(BASE58_ALPHABET[b58[i]]);
  return result;
}

void hash160(const uint8_t* data, size_t len, std::array<uint8_t,20>& out) {
  uint8_t h1[32];
  sha256_once(data, len, h1);
  RIPEMD160(h1, 32, out.data());
}

bool privkey_to_compressed_pubkey(const uint8_t privkey[32], std::array<uint8_t,33>& out_pubkey, secp256k1_context* ctx) {
  if (!secp256k1_ec_seckey_verify(ctx, privkey)) return false;
  secp256k1_pubkey pubkey;
  if (!secp256k1_ec_pubkey_create(ctx, &pubkey, privkey)) return false;
  size_t outlen = 33;
  if (!secp256k1_ec_pubkey_serialize(ctx, out_pubkey.data(), &outlen, &pubkey, SECP256K1_EC_COMPRESSED)) return false;
  return outlen == 33;
}

void mpz_to_32be(mpz_t v, uint8_t out[32]) {
  std::memset(out, 0, 32);
  size_t count = 0;
  (void)mpz_export(nullptr, &count, 1, 1, 1, 0, v);
  std::vector<uint8_t> buf(count);
  size_t count2 = 0;
  mpz_export(buf.data(), &count2, 1, 1, 1, 0, v);
  if (count2 > 32) {
    std::memcpy(out, buf.data() + (count2 - 32), 32);
  } else {
    std::memcpy(out + (32 - count2), buf.data(), count2);
  }
}

std::string mpz_to_hex(const mpz_t v) {
  char* s = mpz_get_str(nullptr, 16, v);
  std::string out(s ? s : "0");
  if (s) free(s);
  return out;
}

std::string bytes_to_hex(const uint8_t* data, size_t len) {
  static const char* hex = "0123456789abcdef";
  std::string out;
  out.resize(len * 2);
  for (size_t i = 0; i < len; ++i) {
    out[2*i] = hex[(data[i] >> 4) & 0xF];
    out[2*i + 1] = hex[data[i] & 0xF];
  }
  return out;
}

int be32_cmp(const uint8_t* a, const uint8_t* b) {
  for (int i = 0; i < 32; ++i) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return 0;
}

void be32_inc(uint8_t* a) {
  for (int i = 31; i >= 0; --i) {
    uint16_t v = (uint16_t)a[i] + 1;
    a[i] = (uint8_t)v;
    if ((v & 0x100) == 0) break;
  }
}


std::string shorten_mid(const std::string& s, size_t keep_front, size_t keep_back) {
    if (s.size() <= keep_front + keep_back) return s;
    return s.substr(0, keep_front) + "..." + s.substr(s.size() - keep_back);
}