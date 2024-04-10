#pragma once
#include<iostream>
namespace mathlib{
    
struct Complex{
    double rl;
    double im;
    constexpr double abs(){ return std::sqrt(rl*rl+im*im); }
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
    constexpr Complex& operator*=(const double factor){
        rl*=factor;
        im*=factor;
        return *this;
    }
    constexpr Complex& operator*=(const Complex& other){
        rl=rl*other.rl-im*other.im;
        im=rl*other.im+im*other.rl;
        return *this;
    }
    constexpr Complex& operator/=(const double factor){
        rl/=factor;
        im/=factor;
        return *this;
    }
    constexpr Complex& operator/=(const Complex& other){
        const double factor{other.rl*other.rl+other.im*other.im};
        rl=(rl*other.rl+im*other.im)/factor;
        im=(rl*other.im-im*other.rl)/factor;
        return *this;
    }
};
constexpr Complex operator+(const Complex& c1, const Complex& c2){ return Complex{c1}+=c2; }
constexpr Complex operator-(const Complex& c1, const Complex& c2){ return Complex{c1}-=c2; }
constexpr Complex operator*(const Complex& c1, const Complex& c2){ return Complex{c1}*=c2; }
constexpr Complex operator/(const Complex& c1, const Complex& c2){ return Complex{c1}/=c2; }
constexpr Complex operator*(const Complex& c, const double factor){ return Complex{c}*=factor; }
constexpr Complex operator/(const Complex& c, const double factor){ return Complex{c}/=factor; }
constexpr Complex operator*(const double factor, const Complex& c){ return Complex{c}*=factor; }
constexpr Complex operator/(const double factor, const Complex& c){ return Complex{c}/=factor; }
inline std::ostream& operator<<(std::ostream& o, const Complex& c){
    if(c.im<0) o<<c.rl<<c.im<<"i";
    else if(c.im>0) o<<c.rl<<"+"<<c.im<<"i";
    else o<<c.rl;
    return o;
}
}