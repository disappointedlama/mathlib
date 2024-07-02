#pragma once
#include"complex_matrix.hpp"
namespace mathlib{
template<size_t size>
class SquareComplexMatrix : public ComplexMatrix<size,size>{
public:
    SquareComplexMatrix(array<Complex,size*size>* data) : ComplexMatrix<size,size>{data}{}
    SquareComplexMatrix() : ComplexMatrix<size,size>{}{}
    SquareComplexMatrix(const SquareComplexMatrix& other) : ComplexMatrix<size,size>{other}{}
    SquareComplexMatrix(const ComplexMatrix<size,size>& other) : ComplexMatrix<size,size>{other}{}
    SquareComplexMatrix(const array<Complex,size*size> data) : ComplexMatrix<size,size>{data}{}
    inline SquareComplexMatrix& transposeInPlace(){   
        #pragma omp parallel for if(size*size>2000)
        for(int i=0;i<size;++i){
            for(int j=i;j<size;++j){
                std::swap((*this->data)[j*size+i], (*this->data)[i*size+j]);
            }
        }
        return *this;
    }
    inline Complex determinant() const {
        return SquareComplexMatrix{*this}.determinantInPlace();
    }
    inline Complex determinantInPlace(){
        array<Complex, size*size>& data{*this->data};
        Complex sign{1};
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
                const Complex factor{data[jOffset+i]/data[iOffset+i]};
                for(int k=i;k<size;++k){
                    data[jOffset+k]-=factor * data[iOffset+k];
                }
            }
        }
        Complex ret{1};
        for(int i=0;i<size;++i){
            ret*=data[i*size+i];
        }
        return sign * ret;
    }
    constexpr bool isOrthogonal() const{
        return (SquareComplexMatrix{*this}.transposeInPlace()*(*this)).isIdentity();
    }
    constexpr bool isIdentity() const{
        for(int i=0;i<ComplexMatrix<size,size>::data->size();++i){
            if(i%size==i/size){
                if(ComplexMatrix<size,size>::data->at(i)!=1) return false;
            }
            else if(ComplexMatrix<size,size>::data->at(i)) return false;
        }
        return true;
    }
    constexpr bool isHermitian() const{
        for(int i=0;i<size;++i){
            for(int j=0;j<size-i;++i){
                if(ComplexMatrix<size,size>::data->at(i*size+j)!=ComplexMatrix<size,size>::data->at(j*size+i).conjugate()) return false;
            }
        }
        return true;
    }
    inline SquareComplexMatrix pow(unsigned int exponent) const{
        SquareComplexMatrix ret{*this};
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
    inline SquareComplexMatrix exponential() const{
        const size_t k{1ULL<<31};
        SquareComplexMatrix m{SquareComplexMatrix::identity()+(1.0/k)*(*this)};
        for(size_t i=1;i<k;i*=2){
            m*=m;
        }
        return m;
    }
    inline ComplexVector<size> solve(ComplexVector<size>& v) const{
        //solve with gauß algorithm
        SquareComplexMatrix<size> m{*this};
        array<Complex, size*size>& data{*m.data};
        for(int i=0;i<size;++i){
            const size_t iOffset{i*size};
            if(data[iOffset+i]==0){
                size_t index{0};
                Complex value{std::numeric_limits<Complex>::min()};
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
                    return ComplexVector<size>::zero();
                }
            }
            for(int j=i+1;j<size;++j){
                const size_t jOffset{j*size};
                const Complex factor{data[jOffset+i]/data[iOffset+i]};
                for(int k=i;k<size;++k){
                    data[jOffset+k]-=factor * data[iOffset+k];
                }
                (*v.data)[j]-=factor * (*v.data)[i];
            }
        }
        ComplexVector<size> ret{};
        for(int i=0;i<size;++i){
            const size_t offset{(size-1-i)*size+size -1};
            Complex rhs{(*v.data)[size-1-i]};
            for(int j=0;j<i;++j){
                rhs-=ret[size-1-j]*data[offset -j];
            }
            ret[size-1-i]=rhs/data[offset -i];
        }
        return ret;
    }
    inline vector<ComplexVector<size>> solve(vector<ComplexVector<size>>& vectors){
        SquareComplexMatrix<size> m{*this};
        array<Complex, size*size>& data{*m.data};
        for(int i=0;i<size;++i){
            const size_t iOffset{i*size};
            if(data[iOffset+i]==0){
                size_t index{0};
                Complex value{std::numeric_limits<Complex>::min()};
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
                    return vector<ComplexVector<size>>{ComplexVector<size>::zero()};
                }
            }
            for(int j=i+1;j<size;++j){
                const size_t jOffset{j*size};
                const Complex factor{data[jOffset+i]/data[iOffset+i]};
                data[jOffset+i]=factor;
                for(int k=i+1;k<size;++k){
                    data[jOffset+k]-=factor * data[iOffset+k];
                }
                (*vectors[0].data)[j]-=factor * (*vectors[0].data)[i];
            }
        }
        for(int i=0;i<size;++i){
            const size_t offset{(size-1-i)*size+size -1};
            Complex rhs{vectors[0][size-1-i]};
            for(int j=0;j<i;++j){
                rhs-=vectors[0][size-1-j]*data[offset -j];
            }
            vectors[0][size-1-i]=rhs/data[offset -i];
        }
        for(int v=1;v<vectors.size();++v){
            for(int i=0;i<size;++i){
                const size_t offset{i*size};
                Complex rhs{vectors[v][i]};
                for(int j=0;j<i;++j){
                    rhs-=vectors[v][j]*data[offset + j];
                }
                vectors[v][i]=rhs;
            }
            for(int i=0;i<size;++i){
                const size_t offset{(size-1-i)*size+size -1};
                Complex rhs{vectors[v][size-1-i]};
                for(int j=0;j<i;++j){
                    rhs-=vectors[v][size-1-j]*data[offset -j];
                }
                vectors[v][size-1-i]=rhs/data[offset -i];
            }
        }
        return vectors;
    }
    inline SquareComplexMatrix cholesky() const{
        //only if matrix is spd
        SquareComplexMatrix ret{*this};
        for(int i=0;i<size;++i){
            const size_t iOffset{i*size+i};
            if(ret[iOffset]<0){
                //matrix not spd
                return -1*SquareComplexMatrix::identity();
            }
            const Complex sqrt{std::sqrt(ret[iOffset])};
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
    static SquareComplexMatrix filledWith(const Complex value){
        SquareComplexMatrix ret{};
        std::fill(ret.data->begin(), ret.data->end(), value);
        return ret;
    }
    static SquareComplexMatrix identity(){
        SquareComplexMatrix ret{};
        for(int i=0;i<size;++i) {
            (*ret.data)[i*size+i]=1;
        }
        return ret;
    }
    template<size_t>
    friend inline SquareComplexMatrix operator*(const SquareComplexMatrix& m1, const SquareComplexMatrix& m2);
    inline SquareComplexMatrix& operator*=(const SquareComplexMatrix& m) {
        array<Complex, size*size>* ret{new array<Complex, size*size>{}};
        array<Complex, size*size>* tmp{new array<Complex, size*size>{}};
        array<Complex, size*size>& arr1{*m.data};
        array<Complex, size*size>& arr2{*tmp};
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
                Complex entry{0};
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
    inline SquareComplexMatrix& operator*=(const Complex factor){
        #pragma omp parallel for if(size*size>=1000000)
        for(int i=0;i<this->data->size();++i){
            (*this->data)[i]*=factor;
        }
        return *this;
    }
};
template<size_t size>
constexpr SquareComplexMatrix<size> operator*(const SquareComplexMatrix<size>& m1, const SquareComplexMatrix<size>& m2) {
    array<Complex, size*size>* ret{new array<Complex, size*size>{}};
    array<Complex, size*size>* tmp{new array<Complex, size*size>{}};
    array<Complex, size*size>& arr1{*m1.data};
    array<Complex, size*size>& arr2{*tmp};
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
            Complex entry{0};
            const size_t jOffset{j*size};
            #pragma omp simd reduction(+:entry)
            for(int k=0;k<size;++k){
                entry+=arr1[iOffset + k] * arr2[jOffset+k];
            }
            (*ret)[i*size + j] = entry;
        }
    }
    delete tmp;
    return SquareComplexMatrix<size>{ret};
}

template<>
inline Complex SquareComplexMatrix<2>::determinant() const {
    return (*this)[0][0]*(*this)[1][1] - (*this)[0][1]*(*this)[1][0];
}
template<>
inline Complex SquareComplexMatrix<2>::determinantInPlace() {
    return determinant();
}
template<>
inline Complex SquareComplexMatrix<3>::determinant() const {
    const array<Complex,9>& arr{(*(this->data.get()))};
    return arr[0]*arr[4]*arr[8] + arr[2]*arr[3]*arr[7] + arr[1]*arr[5]*arr[6] - arr[2]*arr[4]*arr[6] - arr[0]*arr[5]*arr[7] - arr[8]*arr[1]*arr[3];
}
template<>
inline Complex SquareComplexMatrix<3>::determinantInPlace() {
    return determinant();
}
}