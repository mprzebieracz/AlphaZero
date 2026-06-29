#ifndef TEST_COMMON_HPP
#define TEST_COMMON_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifdef CHECK
#undef CHECK
#endif

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "[FAIL] %s:%d: CHECK(%s)\n", __FILE__,        \
                         __LINE__, #cond);                                     \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        auto _aa = (a);                                                        \
        auto _bb = (b);                                                        \
        if (!(_aa == _bb)) {                                                   \
            std::fprintf(stderr,                                               \
                         "[FAIL] %s:%d: CHECK_EQ(%s, %s)\n",                   \
                         __FILE__, __LINE__, #a, #b);                          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
    do {                                                                       \
        double _aa = (a);                                                      \
        double _bb = (b);                                                      \
        double _ee = (eps);                                                    \
        if (!(std::fabs(_aa - _bb) <= _ee)) {                                  \
            std::fprintf(stderr,                                               \
                         "[FAIL] %s:%d: CHECK_NEAR(%s=%g, %s=%g, eps=%g)\n",   \
                         __FILE__, __LINE__, #a, _aa, #b, _bb, _ee);           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define RUN(name)                                                              \
    do {                                                                       \
        std::printf("[RUN ] %s\n", #name);                                     \
        name();                                                                \
        std::printf("[ OK ] %s\n", #name);                                     \
    } while (0)

#endif // TEST_COMMON_HPP
