#pragma once
#include<random>
#include<chrono>
#include<array>
#include<memory>
#include<cassert>
#include"complex.hpp"
#include"real_matrix.hpp"
namespace mathlib{
using std::array;
template<size_t rows, size_t cols>
struct ComplexMatrix{
    class ComplexMatrixRow{
        static const size_t size = cols;
        Complex* start;
    public:
        constexpr ComplexMatrixRow(Complex* start): start{start}{}
        constexpr Complex& operator[](const size_t pos) {
            assert(pos<size);
            return start[pos];
        }
        constexpr const Complex& operator[](const size_t pos)const {
            assert(pos<size);
            return start[pos];
        }
    };
    static std::default_random_engine re;
public:
    std::unique_ptr<array<Complex,rows*cols>> data;
    ComplexMatrix() : data{new array<Complex,rows*cols>{}}{}
    ComplexMatrix(array<Complex,rows*cols>* data) : data{data}{}
    ComplexMatrix(const ComplexMatrix& other) : data{new array<Complex,rows*cols>{*other.data}}{}
    ComplexMatrix(const array<Complex,rows*cols> data) : data{new array<Complex,rows*cols>{data}}{}
    static ComplexMatrix filledWith(const Complex value){
        ComplexMatrix ret{};
        std::fill(ret.data->begin(), ret.data->end(), value);
        return ret;
    }
    inline ComplexMatrix& swapRows(const size_t i, const size_t j){
        const size_t iOffset{i*cols};
        const size_t jOffset{j*cols};
        for(int k=0;k<cols;++k){
            std::swap((*data)[iOffset+k],(*data)[jOffset+k]);
        }
        return *this;
    }
    inline ComplexMatrix& swapCols(const size_t i, const size_t j){
        for(int k=0;k<rows;++k){
            const size_t offset{k*cols};
            std::swap((*data)[offset+i],(*data)[offset+j]);
        }
        return *this;
    }
    inline ComplexMatrix<cols,rows> transpose() const{
        array<Complex, cols*rows>* ret{ new array<Complex, cols*rows>{} };        
        #pragma omp parallel for if(rows*cols>2000)
        for(int i=0;i<cols;++i){
            for(int j=0;j<rows;++j){
                (*ret)[j*cols+i]=(*data)[i*cols+j];
            }
        }
        return ComplexMatrix<cols,rows>{ret};
    }
    static ComplexMatrix getRandom(const Complex lower_bound, const Complex upper_bound){
        std::uniform_real_distribution<long double> unif(lower_bound,upper_bound);
        ComplexMatrix ret{};
        for(int i=0;i<ret.data->size();++i){
            ret[i]=Complex{unif(ComplexMatrix::re),unif(ComplexMatrix::re)};
        }
        return ret;
    }
    inline ComplexMatrix& operator=(const ComplexMatrix& other){
        data=std::make_unique<array<Complex,rows*cols>>(*other.data);
        return *this;
    }
    inline ComplexMatrix& operator+=(const ComplexMatrix& rhs){
        #pragma omp parallel for if(rows*cols>=1000000)
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[cols*i+j]+=(*rhs.data)[cols*i+j];
            }
        }
        return *this;
    }
    inline ComplexMatrix& operator-=(const ComplexMatrix& rhs){
        #pragma omp parallel for if(rows*cols>=1000000)
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[cols*i+j]-=(*rhs.data)[cols*i+j];
            }
        }
        return *this;
    }
    inline ComplexMatrix& operator*=(const Complex factor){
        #pragma omp parallel for if(rows*cols>=1000000)
        for(int i=0;i<data->size();++i){
            (*data)[i]*=factor;
        }
        return *this;
    }
    constexpr ComplexMatrixRow operator[](const size_t pos){
        assert(pos<rows);
        return ComplexMatrixRow{(Complex*)data.get()+pos*cols};
    }
    constexpr const ComplexMatrixRow operator[](const size_t pos) const{
        assert(pos<rows);
        return ComplexMatrixRow{(Complex*)data.get()+pos*cols};
    }
    friend constexpr bool operator==(const ComplexMatrix& m1, const ComplexMatrix& m2){ return !(m1!=m2); }
    friend constexpr bool operator!=(const ComplexMatrix& m1, const ComplexMatrix& m2){ return *m1.data!=*m2.data; }
    template<typename T>
    friend inline ComplexMatrix operator*(const T& factor, const ComplexMatrix& m){
        ComplexMatrix ret{m};
        ret*=factor;
        return ret;
    }
    template<typename T>
    friend inline ComplexMatrix operator*(const ComplexMatrix& m, const T& factor){
        return factor*m;
    }
    template<size_t other_cols>
    friend inline ComplexMatrix<rows,other_cols> operator*(const ComplexMatrix<rows,cols>& m1, const ComplexMatrix<cols,other_cols>& m2) {
        array<Complex, rows*other_cols>* ret{new array<Complex, rows*other_cols>{}};
        std::unique_ptr<array<Complex, other_cols*cols>> tmp=std::make_unique<array<Complex, other_cols*cols>>();
        array<Complex, rows*cols>& arr1{*m1.data};
        array<Complex, other_cols*cols>& arr2{*tmp};
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
                Complex entry{0};
                const size_t jOffset{j*cols};
                #pragma omp simd reduction(+:entry)
                for(int k=0;k<cols;++k){
                    entry+=arr1[iOffset + k] * arr2[jOffset+k];
                }
                (*ret)[i*other_cols + j] = entry;
            }
        }
        return ComplexMatrix<rows,other_cols>{ret};
    }
    template<size_t other_cols>
    friend inline ComplexMatrix<rows,other_cols> operator*(const ComplexMatrix<rows,cols>& m1, const RealMatrix<cols,other_cols>& m2) {
        array<Complex, rows*other_cols>* ret{new array<Complex, rows*other_cols>{}};
        std::unique_ptr<array<long double, other_cols*cols>> tmp=std::make_unique<array<long double, other_cols*cols>>();
        array<Complex, rows*cols>& arr1{*m1.data};
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
                Complex entry{0};
                const size_t jOffset{j*cols};
                #pragma omp simd reduction(+:entry)
                for(int k=0;k<cols;++k){
                    entry+=arr1[iOffset + k] * arr2[jOffset+k];
                }
                (*ret)[i*other_cols + j] = entry;
            }
        }
        return ComplexMatrix<rows,other_cols>{ret};
    }
    friend std::ostream& operator<<(std::ostream& o, const ComplexMatrix& m){
        for(int i=0;i<m.data->size();++i){
            if((i)%cols==0 && i) o<<"\n";
            o<<std::setw(6)<<(*m.data)[i]<<" ";
        }
        return o<<"\n";
    }
    template<typename T>
    inline void apply(T (*func)(Complex)){
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[i*cols+j] = static_cast<Complex>(func((*data)[i*cols+j]));
            }
        }
    }
    template<typename T>
    inline void apply(T (*func)(const size_t, const size_t)){
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[i*cols+j] = static_cast<Complex>(func(i,j));
            }
        }
    }
    template<typename T>
    inline void apply(T (*func)(const size_t, const size_t, Complex)){
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[i*cols+j] = static_cast<Complex>(func(i,j,(*data)[i*cols+j]));
            }
        }
    }
};

template<size_t rows, size_t cols> std::default_random_engine ComplexMatrix<rows,cols>::re{(unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count()};
}