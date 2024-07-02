#pragma once
#include"complex_matrix.hpp"
namespace mathlib{
template<size_t dim>
class ComplexVector : public ComplexMatrix<dim,1>{
public:
    ComplexVector(array<Complex,dim>* data) : ComplexMatrix<dim,1>{data}{}
    ComplexVector() : ComplexMatrix<dim,1>{}{}
    ComplexVector(const ComplexVector& other) : ComplexMatrix<dim,1>{other}{}
    ComplexVector(const ComplexMatrix<dim,1>& other) : ComplexMatrix<dim,1>{other}{}
    ComplexVector(const array<Complex,dim> data) : ComplexMatrix<dim,1>{data}{}
    inline ComplexVector& operator*=(const long double factor){
        for(int i=0;i<this->data->size();++i){
            (*this->data)[i]*=factor;
        }
        return *this;
    }
    inline ComplexVector& operator*=(const Complex factor){
        for(int i=0;i<this->data->size();++i){
            (*this->data)[i]*=factor;
        }
        return *this;
    }
    constexpr Complex& operator[](const size_t pos){
        assert(pos<dim);
        return *((Complex*)ComplexMatrix<dim,1>::data.get()+pos);
    }
    constexpr const Complex& operator[](const size_t pos) const{
        assert(pos<dim);
        return *((Complex*)ComplexMatrix<dim,1>::data.get()+pos);
    }
    friend constexpr Complex operator*(const ComplexVector<dim>& lhs, const ComplexVector<dim>& rhs){
        Complex ret{};
        const array<Complex,dim>& arr1{*lhs.data};
        const array<Complex,dim>& arr2{*rhs.data};
        #pragma omp parallel for simd reduction(+:ret) if(dim>100000)
        for(int i=0;i<dim;++i){
            ret+=arr1[i]*arr2[i];
        }
        return ret;
    }
    constexpr Complex abs() const{
        Complex ret{};
        const array<Complex,dim>& data{*ComplexMatrix<dim,1>::data};
        #pragma omp parallel for simd reduction(+:ret) if(dim>100000)
        for(int i=0;i<dim;++i){
            ret+=data[i]*data[i];
        }
        return ret.abs();
    }
    static inline ComplexVector zero(){
        array<Complex,dim>* data{};
        for(int i=0;i<dim;++i){
            (*data)[i]=0;
        }
        return ComplexVector{data};
    }
};
inline ComplexVector<3> crossProduct(const ComplexVector<3>& v1, const ComplexVector<3>& v2){
    return ComplexVector<3>{{v1[1]*v2[2]-v1[2]*v2[1],v1[2]*v2[0]-v1[0]*v2[2],v1[0]*v2[1]-v1[1]*v2[0]}};
}
template<size_t dim>
constexpr Complex angle(const ComplexVector<dim>& v1, const ComplexVector<dim>& v2){
    return std::acos((v1*v2)/(v1.abs()*v2.abs()));
}
}