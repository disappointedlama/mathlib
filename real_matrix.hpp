#pragma once
#include<random>
#include<chrono>
#include<array>
#include<iostream>
#include<iomanip>
#include<memory>
#include<cassert>
#include<type_traits>
#include"omp.h"
namespace mathlib{
using std::array, std::vector, std::cout, std::endl;
template<size_t rows, size_t cols>
class RealMatrix{
    class RealMatrixRow{
        static const size_t size = cols;
        long double* start;
    public:
        constexpr RealMatrixRow(long double* start): start{start}{}
        constexpr long double& operator[](const size_t pos) {
            assert(pos<size);
            return start[pos];
        }
        constexpr const long double& operator[](const size_t pos)const {
            assert(pos<size);
            return start[pos];
        }
    };
    static std::default_random_engine re;
public:
    std::unique_ptr<array<long double,rows*cols>> data;
    RealMatrix() : data{new array<long double,rows*cols>{}}{}
    RealMatrix(array<long double,rows*cols>* data) : data{data}{}
    RealMatrix(const RealMatrix& other) : data{new array<long double,rows*cols>{*other.data}}{}
    RealMatrix(const array<long double,rows*cols> data) : data{new array<long double,rows*cols>{data}}{}
    static RealMatrix filledWith(const long double value){
        RealMatrix ret{};
        std::fill(ret.data->begin(), ret.data->end(), value);
        return ret;
    }
    inline RealMatrix& swapRows(const size_t i, const size_t j){
        const size_t iOffset{i*cols};
        const size_t jOffset{j*cols};
        for(int k=0;k<cols;++k){
            std::swap((*data)[iOffset+k],(*data)[jOffset+k]);
        }
        return *this;
    }
    inline RealMatrix& swapCols(const size_t i, const size_t j){
        for(int k=0;k<rows;++k){
            const size_t offset{k*cols};
            std::swap((*data)[offset+i],(*data)[offset+j]);
        }
        return *this;
    }
    inline RealMatrix<cols,rows> transpose() const{
        array<long double, cols*rows>* ret{ new array<long double, cols*rows>{} };        
        #pragma omp parallel for if(rows*cols>2000)
        for(int i=0;i<cols;++i){
            for(int j=0;j<rows;++j){
                (*ret)[j*cols+i]=(*data)[i*cols+j];
            }
        }
        return RealMatrix<cols,rows>{ret};
    }
    static RealMatrix getRandom(const long double lower_bound, const long double upper_bound){
        std::uniform_real_distribution<long double> unif(lower_bound,upper_bound);
        RealMatrix ret{};
        for(int i=0;i<ret.data->size();++i){
            ret[i]=unif(RealMatrix::re);
        }
        return ret;
    }
    inline RealMatrix& operator=(const RealMatrix& other){
        data=std::make_unique<array<long double, rows*cols>>(*other.data);
        return *this;
    }
    inline RealMatrix& operator+=(const RealMatrix& rhs){
        #pragma omp parallel for if(rows*cols>=1000000)
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[cols*i+j]+=(*rhs.data)[cols*i+j];
            }
        }
        return *this;
    }
    inline RealMatrix& operator-=(const RealMatrix& rhs){
        #pragma omp parallel for if(rows*cols>=1000000)
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[cols*i+j]-=(*rhs.data)[cols*i+j];
            }
        }
        return *this;
    }
    inline RealMatrix& operator*=(const long double factor){
        #pragma omp parallel for if(rows*cols>=1000000)
        for(int i=0;i<data->size();++i){
            (*data)[i]*=factor;
        }
        return *this;
    }
    constexpr RealMatrixRow operator[](const size_t pos){
        assert(pos<rows);
        return RealMatrixRow{(long double*)data.get()+pos*cols};
    }
    constexpr const RealMatrixRow operator[](const size_t pos) const{
        assert(pos<rows);
        return RealMatrixRow{(long double*)data.get()+pos*cols};
    }
    friend constexpr bool operator==(const RealMatrix& m1, const RealMatrix& m2){ return !(m1!=m2); }
    friend constexpr bool operator!=(const RealMatrix& m1, const RealMatrix& m2){ return *m1.data!=*m2.data; }
    template<size_t other_cols>
    friend inline RealMatrix<rows,other_cols> operator*(const RealMatrix<rows,cols>& m1, const RealMatrix<cols,other_cols>& m2) {
        array<long double, rows*other_cols>* ret{new array<long double, rows*other_cols>{}};
        std::unique_ptr<array<long double, other_cols*cols>> tmp=std::make_unique<array<long double, other_cols*cols>>();
        array<long double, rows*cols>& arr1{*m1.data};
        array<long double, other_cols*cols>& arr2{*tmp};
        #pragma omp parallel for if(other_cols*cols>2000)
        for(int i=0;i<cols;++i){
            const size_t iOffset{i*other_cols};
            for(int j=0;j<other_cols;++j){
                arr2[j*other_cols+i]=(*m2.data)[iOffset+j];
            }
        }
        #pragma omp parallel for if(rows*cols>2000 || other_cols*cols>2000)
        for(int i=0;i<rows;++i){
            const size_t iOffset{i*cols};
            for(int j=0;j<other_cols;++j){
                long double entry{0};
                const size_t jOffset{j*cols};
                #pragma omp simd reduction(+:entry)
                for(int k=0;k<cols;++k){
                    entry+=arr1[iOffset + k] * arr2[jOffset+k];
                }
                (*ret)[i*other_cols + j] = entry;
            }
        }
        return RealMatrix<rows,other_cols>{ret};
    }
    friend std::ostream& operator<<(std::ostream& o, const RealMatrix& m){
        for(int i=0;i<m.data->size();++i){
            if((i)%cols==0 && i) o<<"\n";
            o<<std::setw(6)<<(*m.data)[i]<<" ";
        }
        return o<<"\n";
    }
    template<typename T>
    inline void apply(T (*func)(long double)){
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[i*cols+j] = static_cast<long double>(func((*data)[i*cols+j]));
            }
        }
    }
    template<typename T>
    inline void apply(T (*func)(const size_t, const size_t)){
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[i*cols+j] = static_cast<long double>(func(i,j));
            }
        }
    }
    template<typename T>
    inline void apply(T (*func)(const size_t, const size_t, long double)){
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[i*cols+j] = static_cast<long double>(func(i,j,(*data)[i*cols+j]));
            }
        }
    }
};
template<size_t rows, size_t cols> std::default_random_engine RealMatrix<rows,cols>::re{(unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count()};
}