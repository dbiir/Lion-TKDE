// benchmark/pps/Schema.h
#pragma once

#include "common/ClassOf.h"
#include "common/FixedString.h"
#include "common/Hash.h"
#include "common/Serialization.h"
#include "core/SchemaDef.h"

namespace star {
namespace pps {
static constexpr auto __BASE_COUNTER__ = __COUNTER__ + 1;
static constexpr auto PPS_FIELD_SIZE = 10;
} // namespace pps
} // namespace star

#undef NAMESPACE_FIELDS
#define NAMESPACE_FIELDS(x) x(star) x(pps)

// Table 0: PARTS
#define PARTS_KEY_FIELDS(x, y) x(int32_t, P_KEY)
#define PARTS_VALUE_FIELDS(x, y)                                               \
  x(int64_t, P_AMOUNT)                                                         \
      y(FixedString<PPS_FIELD_SIZE>, P_F01)                                    \
          y(FixedString<PPS_FIELD_SIZE>, P_F02)                                \
              y(FixedString<PPS_FIELD_SIZE>, P_F03)                            \
                  y(FixedString<PPS_FIELD_SIZE>, P_F04)                        \
                      y(FixedString<PPS_FIELD_SIZE>, P_F05)                    \
                          y(FixedString<PPS_FIELD_SIZE>, P_F06)                \
                              y(FixedString<PPS_FIELD_SIZE>, P_F07)            \
                                  y(FixedString<PPS_FIELD_SIZE>, P_F08)        \
                                      y(FixedString<PPS_FIELD_SIZE>, P_F09)    \
                                          y(FixedString<PPS_FIELD_SIZE>, P_F10)

DO_STRUCT(parts, PARTS_KEY_FIELDS, PARTS_VALUE_FIELDS, NAMESPACE_FIELDS)

// Table 1: PRODUCTS
#define PRODUCTS_KEY_FIELDS(x, y) x(int32_t, PR_KEY)
#define PRODUCTS_VALUE_FIELDS(x, y)                                            \
  x(FixedString<PPS_FIELD_SIZE>, PR_F01)                                       \
      y(FixedString<PPS_FIELD_SIZE>, PR_F02)                                   \
          y(FixedString<PPS_FIELD_SIZE>, PR_F03)                               \
              y(FixedString<PPS_FIELD_SIZE>, PR_F04)                           \
                  y(FixedString<PPS_FIELD_SIZE>, PR_F05)                       \
                      y(FixedString<PPS_FIELD_SIZE>, PR_F06)                   \
                          y(FixedString<PPS_FIELD_SIZE>, PR_F07)               \
                              y(FixedString<PPS_FIELD_SIZE>, PR_F08)           \
                                  y(FixedString<PPS_FIELD_SIZE>, PR_F09)       \
                                      y(FixedString<PPS_FIELD_SIZE>, PR_F10)

DO_STRUCT(products, PRODUCTS_KEY_FIELDS, PRODUCTS_VALUE_FIELDS, NAMESPACE_FIELDS)

// Table 2: SUPPLIERS
#define SUPPLIERS_KEY_FIELDS(x, y) x(int32_t, S_KEY)
#define SUPPLIERS_VALUE_FIELDS(x, y)                                           \
  x(FixedString<PPS_FIELD_SIZE>, S_F01)                                        \
      y(FixedString<PPS_FIELD_SIZE>, S_F02)                                    \
          y(FixedString<PPS_FIELD_SIZE>, S_F03)                                \
              y(FixedString<PPS_FIELD_SIZE>, S_F04)                            \
                  y(FixedString<PPS_FIELD_SIZE>, S_F05)                        \
                      y(FixedString<PPS_FIELD_SIZE>, S_F06)                    \
                          y(FixedString<PPS_FIELD_SIZE>, S_F07)                \
                              y(FixedString<PPS_FIELD_SIZE>, S_F08)            \
                                  y(FixedString<PPS_FIELD_SIZE>, S_F09)        \
                                      y(FixedString<PPS_FIELD_SIZE>, S_F10)

DO_STRUCT(suppliers, SUPPLIERS_KEY_FIELDS, SUPPLIERS_VALUE_FIELDS, NAMESPACE_FIELDS)

// After 3 DO_STRUCT calls:
//   parts::tableID    == 0
//   products::tableID == 1
//   suppliers::tableID == 2

namespace star {

// parts: only P_AMOUNT is the mutable field for serialization
template <> class Serializer<pps::parts::value> {
public:
  std::string operator()(const pps::parts::value &v) {
    return Serializer<decltype(v.P_AMOUNT)>()(v.P_AMOUNT);
  }
};
template <> class Deserializer<pps::parts::value> {
public:
  std::size_t operator()(StringPiece str, pps::parts::value &result) const {
    return Deserializer<decltype(result.P_AMOUNT)>()(str, result.P_AMOUNT);
  }
};
template <> class ClassOf<pps::parts::value> {
public:
  static constexpr std::size_t size() {
    return ClassOf<decltype(pps::parts::value::P_AMOUNT)>::size();
  }
};

// products: PR_F01 is the mutable field
template <> class Serializer<pps::products::value> {
public:
  std::string operator()(const pps::products::value &v) {
    return Serializer<decltype(v.PR_F01)>()(v.PR_F01);
  }
};
template <> class Deserializer<pps::products::value> {
public:
  std::size_t operator()(StringPiece str, pps::products::value &result) const {
    return Deserializer<decltype(result.PR_F01)>()(str, result.PR_F01);
  }
};
template <> class ClassOf<pps::products::value> {
public:
  static constexpr std::size_t size() {
    return ClassOf<decltype(pps::products::value::PR_F01)>::size();
  }
};

// suppliers: S_F01 is the mutable field
template <> class Serializer<pps::suppliers::value> {
public:
  std::string operator()(const pps::suppliers::value &v) {
    return Serializer<decltype(v.S_F01)>()(v.S_F01);
  }
};
template <> class Deserializer<pps::suppliers::value> {
public:
  std::size_t operator()(StringPiece str, pps::suppliers::value &result) const {
    return Deserializer<decltype(result.S_F01)>()(str, result.S_F01);
  }
};
template <> class ClassOf<pps::suppliers::value> {
public:
  static constexpr std::size_t size() {
    return ClassOf<decltype(pps::suppliers::value::S_F01)>::size();
  }
};

} // namespace star
