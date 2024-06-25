#pragma once
#include<iostream>
namespace mathlib{
struct Complex;
constexpr Complex operator ""_i(long double arg);
struct Complex{
    long double rl;
    long double im;
    constexpr Complex() : rl{0}, im{0}{}
    constexpr Complex(const long double rl, const long double im) : rl{rl}, im{im}{}
    constexpr Complex(const long double rl) : rl{rl}, im{0}{}
    constexpr long double abs(){ return std::sqrt(rl*rl+im*im); }
    constexpr Complex& conjugateInPlace(){ im*=-1; return *this; }
    constexpr Complex conjugate(){ return Complex{rl,-im}; }
    constexpr Complex& operator+=(const Complex& other){
        rl+=other.rl;
        im+=other.im;
        return *this;
    };
    constexpr Complex& operator-=(const Complex& other){
        rl-=other.rl;
        im-=other.im;
        return *this;
    };
    constexpr Complex& operator*=(const long double factor){
        rl*=factor;
        im*=factor;
        return *this;
    }
    constexpr Complex& operator*=(const Complex& other){
        rl=rl*other.rl-im*other.im;
        im=rl*other.im+im*other.rl;
        return *this;
    }
    constexpr Complex& operator/=(const long double factor){
        rl/=factor;
        im/=factor;
        return *this;
    }
    constexpr Complex& operator/=(const Complex& other){
        const long double factor{other.rl*other.rl+other.im*other.im};
        rl=(rl*other.rl+im*other.im)/factor;
        im=(rl*other.im-im*other.rl)/factor;
        return *this;
    }
};
constexpr Complex operator+(const Complex& c1, const Complex& c2){ return Complex{c1}+=c2; }
constexpr Complex operator-(const Complex& c1, const Complex& c2){ return Complex{c1}-=c2; }
constexpr Complex operator*(const Complex& c1, const Complex& c2){ return Complex{c1}*=c2; }
constexpr Complex operator/(const Complex& c1, const Complex& c2){ return Complex{c1}/=c2; }
constexpr Complex operator*(const Complex& c, const long double factor){ return Complex{c}*=factor; }
constexpr Complex operator/(const Complex& c, const long double factor){ return Complex{c}/=factor; }
template<typename T>
constexpr Complex operator+(const T factor, const Complex& c){ return Complex{c}+=factor; }
template<typename T>
constexpr Complex operator-(const T factor, const Complex& c){ return Complex{c}-=factor; }
template<typename T>
constexpr Complex operator*(const T factor, const Complex& c){ return Complex{c}*=factor; }
template<typename T>
constexpr Complex operator/(const T factor, const Complex& c){ return Complex{c}/=factor; }
constexpr Complex operator*(const long double factor, const Complex& c){ return Complex{c}*=factor; }
constexpr Complex operator/(const long double factor, const Complex& c){ return Complex{c}/=factor; }
inline std::ostream& operator<<(std::ostream& o, const Complex& c){
    if(c.im<0) o<<c.rl<<c.im<<"i";
    else if(c.im>0) o<<c.rl<<"+"<<c.im<<"i";
    else o<<c.rl;
    return o;
}
constexpr Complex operator ""_i(long double arg){
    return Complex{0,arg};
}
}