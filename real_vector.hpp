#pragma once
#include"real_matrix.hpp"
namespace mathlib{
template<size_t dim>
class RealVector : public RealMatrix<dim,1>{
public:
    RealVector(array<long double,dim>* data) : RealMatrix<dim,1>{data}{}
    RealVector() : RealMatrix<dim,1>{}{}
    RealVector(const RealVector& other) : RealMatrix<dim,1>{other}{}
    RealVector(const RealMatrix<dim,1>& other) : RealMatrix<dim,1>{other}{}
    RealVector(const array<long double,dim> data) : RealMatrix<dim,1>{data}{}
    inline RealVector& operator*=(const long double factor){
        for(int i=0;i<this->data->size();++i){
            (*this->data)[i]*=factor;
        }
        return *this;
    }
    friend constexpr long double operator*(const RealVector<dim>& lhs, const RealVector<dim>& rhs){
        long double ret{};
        const array<long double,dim>& arr1{*lhs.data};
        const array<long double,dim>& arr2{*rhs.data};
        #pragma omp parallel for simd reduction(+:ret) if(dim>100000)
        for(int i=0;i<dim;++i){
            ret+=arr1[i]*arr2[i];
        }
        return ret;
    }
    constexpr long double abs() const{
        long double ret{};
        const array<long double,dim>& data{*RealMatrix<dim,1>::data};
        #pragma omp parallel for simd reduction(+:ret) if(dim>100000)
        for(int i=0;i<dim;++i){
            ret+=data[i]*data[i];
        }
        return std::sqrt(ret);
    }
    static inline RealVector zero(){
        array<long double,dim>* data{};
        for(int i=0;i<dim;++i){
            (*data)[i]=0;
        }
        return RealVector{data};
    }
};
inline RealVector<3> crossProduct(const RealVector<3>& v1, const RealVector<3>& v2){
    return RealVector<3>{{v1[1]*v2[2]-v1[2]*v2[1],v1[2]*v2[0]-v1[0]*v2[2],v1[0]*v2[1]-v1[1]*v2[0]}};
}
template<size_t dim>
constexpr long double angle(const RealVector<dim>& v1, const RealVector<dim>& v2){
    return std::acos((v1*v2)/(v1.abs()*v2.abs()));
}
}