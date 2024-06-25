#pragma once
#include"real_matrix.hpp"
namespace mathlib{
template<size_t size>
class SquareRealMatrix : public RealMatrix<size,size>{
public:
    SquareRealMatrix(array<long double,size*size>* data) : RealMatrix<size,size>{data}{}
    SquareRealMatrix() : RealMatrix<size,size>{}{}
    SquareRealMatrix(const SquareRealMatrix& other) : RealMatrix<size,size>{other}{}
    SquareRealMatrix(const RealMatrix<size,size>& other) : RealMatrix<size,size>{other}{}
    SquareRealMatrix(const array<long double,size*size> data) : RealMatrix<size,size>{data}{}
    inline SquareRealMatrix& transposeInPlace(){   
        #pragma omp parallel for if(size*size>2000)
        for(int i=0;i<size;++i){
            for(int j=i;j<size;++j){
                std::swap((*this->data)[j*size+i], (*this->data)[i*size+j]);
            }
        }
        return *this;
    }
    constexpr long double determinant() const {
        return SquareRealMatrix{*this}.determinantInPlace();
    }
    constexpr long double determinantInPlace(){
        array<long double, size*size>& data{*this->data};
        long double sign{1};
        for(int i=0;i<size;++i){
            const size_t iOffset{i*size};
            if(data[iOffset+i]==0){
                bool found{false};
                for(int j=i+1;j<size;++j){
                    if(data[j*size+i]!=0){
                        this->swapRows(i,j);
                        sign*=-1;
                        found=true;
                        break;
                    }
                }
                if(!found){
                    return 0;
                }
            }
            for(int j=i+1;j<size;++j){
                const size_t jOffset{j*size};
                const long double factor{data[jOffset+i]/data[iOffset+i]};
                for(int k=i;k<size;++k){
                    data[jOffset+k]-=factor * data[iOffset+k];
                }
            }
        }
        long double ret{1};
        for(int i=0;i<size;++i){
            ret*=data[i*size+i];
        }
        return sign * ret;
    }
    constexpr bool isOrthogonal() const{
        return (SquareRealMatrix{*this}.transposeInPlace()*(*this)).isIdentity();
    }
    constexpr bool isIdentity() const{
        for(int i=0;i<RealMatrix<size,size>::data->size();++i){
            if(i%size==i/size){
                if(RealMatrix<size,size>::data->at(i)!=1) return false;
            }
            else if(RealMatrix<size,size>::data->at(i)) return false;
        }
        return true;
    }
    constexpr bool isSymmetric() const{
        for(int i=0;i<size;++i){
            for(int j=0;j<size-i;++i){
                if(RealMatrix<size,size>::data->at(i*size+j)!=RealMatrix<size,size>::data->at(j*size+i)) return false;
            }
        }
        return true;
    }
    constexpr bool isAntiSymmetric() const{
        for(int i=0;i<size;++i){
            for(int j=0;j<size-i;++i){
                if(RealMatrix<size,size>::data->at(i*size+j)!=-RealMatrix<size,size>::data->at(j*size+i)) return false;
            }
        }
        return true;
    }
    constexpr bool isSpd() const{
        return cholesky()!= -1 * identity();
    }
    inline SquareRealMatrix pow(unsigned int exponent) const{
        SquareRealMatrix ret{*this};
        size_t powersOfTwo{0};
        unsigned int tmp{1};
        while((tmp*=2)<exponent){
            ++powersOfTwo;
        }
        tmp/=2;
        for(int i=0;i<powersOfTwo;++i){
            ret*=ret;
        }
        for(int i=0;i<exponent-tmp;++i){
            ret*=*this;
        }
        return ret;
    }
    inline SquareRealMatrix exponential() const{
        const size_t k{1ULL<<31};
        SquareRealMatrix m{SquareRealMatrix::identity()+(1.0/k)*(*this)};
        for(size_t i=1;i<k;i*=2){
            m*=m;
        }
        return m;
    }
    inline RealVector<size> solve(RealVector<size>& v) const{
        //solve with gauß algorithm
        SquareRealMatrix<size> m{*this};
        array<long double, size*size>& data{*m.data};
        for(int i=0;i<size;++i){
            const size_t iOffset{i*size};
            if(data[iOffset+i]==0){
                size_t index{0};
                long double value{std::numeric_limits<long double>::min()};
                for(int j=i+1;j<size;++j){
                    if(data[j*size+i]!=0 && std::abs(data[j*size+i])>value){
                        index=j;
                        value=std::abs(data[j*size+i]);
                    }
                }
                if(index){
                    m.swapRows(i,index);
                    v.swapRows(i,index);
                }
                else{
                    return RealVector<size>::zero();
                }
            }
            for(int j=i+1;j<size;++j){
                const size_t jOffset{j*size};
                const long double factor{data[jOffset+i]/data[iOffset+i]};
                for(int k=i;k<size;++k){
                    data[jOffset+k]-=factor * data[iOffset+k];
                }
                (*v.data)[j]-=factor * (*v.data)[i];
            }
        }
        RealVector<size> ret{};
        for(int i=0;i<size;++i){
            const size_t offset{(size-1-i)*size+size -1};
            long double rhs{(*v.data)[size-1-i]};
            for(int j=0;j<i;++j){
                rhs-=ret[size-1-j]*data[offset -j];
            }
            ret[size-1-i]=rhs/data[offset -i];
        }
        return ret;
    }
    inline vector<RealVector<size>> solve(vector<RealVector<size>>& vectors){
        SquareRealMatrix<size> m{*this};
        array<long double, size*size>& data{*m.data};
        for(int i=0;i<size;++i){
            const size_t iOffset{i*size};
            if(data[iOffset+i]==0){
                size_t index{0};
                long double value{std::numeric_limits<long double>::min()};
                for(int j=i+1;j<size;++j){
                    if(data[j*size+i]!=0 && std::abs(data[j*size+i])>value){
                        index=j;
                        value=std::abs(data[j*size+i]);
                    }
                }
                if(index){
                    m.swapRows(i,index);
                    for(auto& v: vectors){
                        v.swapRows(i,index);
                    }
                }
                else{
                    return vector<RealVector<size>>{RealVector<size>::zero()};
                }
            }
            for(int j=i+1;j<size;++j){
                const size_t jOffset{j*size};
                const long double factor{data[jOffset+i]/data[iOffset+i]};
                data[jOffset+i]=factor;
                for(int k=i+1;k<size;++k){
                    data[jOffset+k]-=factor * data[iOffset+k];
                }
                (*vectors[0].data)[j]-=factor * (*vectors[0].data)[i];
            }
        }
        for(int i=0;i<size;++i){
            const size_t offset{(size-1-i)*size+size -1};
            long double rhs{vectors[0][size-1-i]};
            for(int j=0;j<i;++j){
                rhs-=vectors[0][size-1-j]*data[offset -j];
            }
            vectors[0][size-1-i]=rhs/data[offset -i];
        }
        for(int v=1;v<vectors.size();++v){
            for(int i=0;i<size;++i){
                const size_t offset{i*size};
                long double rhs{vectors[v][i]};
                for(int j=0;j<i;++j){
                    rhs-=vectors[v][j]*data[offset + j];
                }
                vectors[v][i]=rhs;
            }
            for(int i=0;i<size;++i){
                const size_t offset{(size-1-i)*size+size -1};
                long double rhs{vectors[v][size-1-i]};
                for(int j=0;j<i;++j){
                    rhs-=vectors[v][size-1-j]*data[offset -j];
                }
                vectors[v][size-1-i]=rhs/data[offset -i];
            }
        }
        return vectors;
    }
    inline SquareRealMatrix cholesky() const{
        //only if matrix is spd
        SquareRealMatrix ret{*this};
        for(int i=0;i<size;++i){
            const size_t iOffset{i*size+i};
            if(ret[iOffset]<0){
                //matrix not spd
                return -1*SquareRealMatrix::identity();
            }
            const long double sqrt{std::sqrt(ret[iOffset])};
            ret[iOffset]=sqrt;
            for(int j=1;j<size-i;++j){
                ret[iOffset+j]/=sqrt;
            }
            for(int j=1;i-j>=0;++j){
                ret[iOffset-j]=0;
            }
            for(int j=i+1;j<size;++j){
                const size_t jOffset{j*size};
                for(int k=j;k<size;++k){
                    ret[jOffset+k]-=ret[i*size+k]*ret[i*size+j];
                }
            }
        }
        return ret.transposeInPlace();
    }
    static SquareRealMatrix filledWith(const long double value){
        SquareRealMatrix ret{};
        std::fill(ret.data->begin(), ret.data->end(), value);
        return ret;
    }
    static SquareRealMatrix identity(){
        SquareRealMatrix ret{};
        for(int i=0;i<size;++i) {
            (*ret.data)[i*size+i]=1;
        }
        return ret;
    }
    template<size_t>
    friend inline SquareRealMatrix operator*(const SquareRealMatrix& m1, const SquareRealMatrix& m2);
    inline SquareRealMatrix& operator*=(const SquareRealMatrix& m) {
        array<long double, size*size>* ret{new array<long double, size*size>{}};
        array<long double, size*size>* tmp{new array<long double, size*size>{}};
        array<long double, size*size>& arr1{*m.data};
        array<long double, size*size>& arr2{*tmp};
        #pragma omp parallel for if(size*size>2000)
        for(int i=0;i<size;++i){
            const size_t iOffset{i*size};
            for(int j=0;j<size;++j){
                arr2[j*size+i]=(*this->data)[iOffset+j];
            }
        }
        #pragma omp parallel for if(size*size>2000)
        for(int i=0;i<size;++i){
            const size_t iOffset{i*size};
            for(int j=0;j<size;++j){
                long double entry{0};
                const size_t jOffset{j*size};
                #pragma omp simd reduction(+:entry)
                for(int k=0;k<size;++k){
                    entry+=arr1[iOffset + k] * arr2[jOffset+k];
                }
                (*ret)[i*size + j] = entry;
            }
        }
        delete tmp;
        delete this->data;
        this->data=ret;
        return *this;
    }
    inline SquareRealMatrix& operator*=(const long double factor){
        #pragma omp parallel for if(size*size>=1000000)
        for(int i=0;i<this->data->size();++i){
            (*this->data)[i]*=factor;
        }
        return *this;
    }
};
template<size_t size>
constexpr SquareRealMatrix<size> operator*(const SquareRealMatrix<size>& m1, const SquareRealMatrix<size>& m2) {
    array<long double, size*size>* ret{new array<long double, size*size>{}};
    array<long double, size*size>* tmp{new array<long double, size*size>{}};
    array<long double, size*size>& arr1{*m1.data};
    array<long double, size*size>& arr2{*tmp};
    #pragma omp parallel for if(size*size>2000)
    for(int i=0;i<size;++i){
        const size_t iOffset{i*size};
        for(int j=0;j<size;++j){
            arr2[j*size+i]=(*m2.data)[iOffset+j];
        }
    }
    #pragma omp parallel for if(size*size>2000)
    for(int i=0;i<size;++i){
        const size_t iOffset{i*size};
        for(int j=0;j<size;++j){
            long double entry{0};
            const size_t jOffset{j*size};
            #pragma omp simd reduction(+:entry)
            for(int k=0;k<size;++k){
                entry+=arr1[iOffset + k] * arr2[jOffset+k];
            }
            (*ret)[i*size + j] = entry;
        }
    }
    delete tmp;
    return SquareRealMatrix<size>{ret};
}

template<>
constexpr long double SquareRealMatrix<2>::determinant() const {
    return (*this)[0]*(*this)[3] - (*this)[1]*(*this)[2];
}
template<>
constexpr long double SquareRealMatrix<2>::determinantInPlace() {
    return determinant();
}
template<>
constexpr long double SquareRealMatrix<3>::determinant() const {
    const array<long double,9>& arr{*this->data};
    return arr[0]*arr[4]*arr[8] + arr[2]*arr[3]*arr[7] + arr[1]*arr[5]*arr[6] - arr[2]*arr[4]*arr[6] - arr[0]*arr[5]*arr[7] - arr[8]*arr[1]*arr[3];
}
template<>
constexpr long double SquareRealMatrix<3>::determinantInPlace() {
    return determinant();
}
}