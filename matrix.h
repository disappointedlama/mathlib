#pragma once
#include<iostream>
#include<iomanip>
#include<array>
#include<vector>
#include<initializer_list>
#include<random>
#include<chrono>
#include"omp.h"
namespace mathlib{
    namespace matrix{
using std::array, std::vector, std::cout, std::endl, std::initializer_list;
template<size_t rows, size_t cols>
class Matrix;
template<size_t dim>
class Vector;
template<size_t rows, size_t cols>
class Matrix{
public:
    static std::default_random_engine re;
    array<double,rows*cols>* data;
    Matrix() : data{new array<double,rows*cols>{}}{}
    Matrix(array<double,rows*cols>* data) : data{data}{}
    Matrix(const Matrix& other) : data{new array<double,rows*cols>{*other.data}}{}
    Matrix(const array<double,rows*cols> data) : data{new array<double,rows*cols>{data}}{}
    ~Matrix(){ delete data; }
    static Matrix filledWith(const double value){
        Matrix ret{};
        std::fill(ret.data->begin(), ret.data->end(), value);
        return ret;
    }
    constexpr Matrix& swapRows(const size_t i, const size_t j){
        const size_t iOffset{i*cols};
        const size_t jOffset{j*cols};
        for(int k=0;k<cols;++k){
            std::swap((*data)[iOffset+k],(*data)[jOffset+k]);
        }
        return *this;
    }
    constexpr Matrix& swapCols(const size_t i, const size_t j){
        for(int k=0;k<rows;++k){
            const size_t offset{k*cols};
            std::swap((*data)[offset+i],(*data)[offset+j]);
        }
        return *this;
    }
    constexpr Matrix<cols,rows> transpose() const{
        array<double, cols*rows>* ret{ new array<double, cols*rows>{} };        
        #pragma omp parallel for if(rows*cols>2000)
        for(int i=0;i<cols;++i){
            for(int j=0;j<rows;++j){
                (*ret)[j*cols+i]=(*data)[i*cols+j];
            }
        }
        return Matrix<cols,rows>{ret};
    }
    static Matrix getRandom(const double lower_bound, const double upper_bound){
        std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
        Matrix ret{};
        for(int i=0;i<ret.data->size();++i){
            ret[i]=unif(Matrix::re);
        }
        return ret;
    }
    constexpr Matrix& operator=(const Matrix& other){
        delete data;
        data=new array<double,rows*cols>{*other.data};
        return *this;
    }
    constexpr Matrix& operator+=(const Matrix& rhs){
        #pragma omp parallel for if(rows*cols>=1000000)
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[cols*i+j]+=(*rhs.data)[cols*i+j];
            }
        }
        return *this;
    }
    constexpr Matrix& operator-=(const Matrix& rhs){
        #pragma omp parallel for if(rows*cols>=1000000)
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                (*data)[cols*i+j]-=(*rhs.data)[cols*i+j];
            }
        }
        return *this;
    }
    constexpr Matrix& operator*=(const double factor){
        #pragma omp parallel for if(rows*cols>=1000000)
        for(int i=0;i<data->size();++i){
            (*data)[i]*=factor;
        }
        return *this;
    }
    friend constexpr Vector<rows> operator*(const Matrix<rows,cols>& m, const Vector<cols>& v){
        array<double,rows> ret{};
        for(int i=0;i<rows;++i){
            double value{0};
            for(int j=0;j<cols;++j){
                value+=(*m.data)[i*cols+j]*(*v.data)[j];
            }
            ret[i]=value;
        }
        return Vector<rows>{ret};
    }
    constexpr double& operator[](size_t pos){ return (*data)[pos]; }
    friend constexpr bool operator==(const Matrix& m1, const Matrix& m2){
        return !(m1!=m2);
    }
    friend constexpr bool operator!=(const Matrix& m1, const Matrix& m2){ return *m1.data!=*m2.data; }
    friend constexpr Matrix<rows,rows> operator*(const Matrix<rows,cols>& m1, const Matrix<cols,rows>& m2) {
        array<double, rows*rows>* ret{new array<double, rows*rows>{}};
        array<double, rows*cols>* tmp{new array<double, rows*cols>{}};
        array<double, rows*cols>& arr1{*m1.data};
        array<double, rows*cols>& arr2{*tmp};
        #pragma omp parallel for if(rows*cols>2000)
        for(int i=0;i<cols;++i){
            const size_t iOffset{i*cols};
            for(int j=0;j<rows;++j){
                arr2[j*cols+i]=(*m2.data)[iOffset+j];
            }
        }
        #pragma omp parallel for if(rows*cols>2000)
        for(int i=0;i<rows;++i){
            const size_t iOffset{i*cols};
            for(int j=0;j<rows;++j){
                double entry{0};
                const size_t jOffset{j*cols};
                #pragma omp simd reduction(+:entry)
                for(int k=0;k<cols;++k){
                    entry+=arr1[iOffset + k] * arr2[jOffset+k];
                }
                (*ret)[i*rows + j] = entry;
            }
        }
        delete tmp;
        return Matrix<rows,rows>{ret};
    }
    friend std::ostream& operator<<(std::ostream& o, const Matrix& m){
        for(int i=0;i<m.data->size();++i){
            if((i)%cols==0 && i) o<<"\n";
            o<<std::setw(6)<<(*m.data)[i]<<" ";
        }
        return o<<"\n";
    }
};
template<size_t rows, size_t cols> std::default_random_engine Matrix<rows,cols>::re{(unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count()};
template<size_t dim>
class Vector : public Matrix<dim,1>{
public:
    Vector(array<double,dim>* data) : Matrix<dim,1>{data}{}
    Vector() : Matrix<dim,1>{}{}
    Vector(const Vector& other) : Matrix<dim,1>{other}{}
    Vector(const Matrix<dim,1>& other) : Matrix<dim,1>{other}{}
    Vector(const array<double,dim> data) : Matrix<dim,1>{data}{}
    constexpr Vector& operator*=(const double factor){
        for(int i=0;i<this->data->size();++i){
            (*this->data)[i]*=factor;
        }
        return *this;
    }
    constexpr double operator*(const Vector<dim>& rhs){
        double ret{};
        const array<double,dim>& arr1{*Matrix<dim,1>::data};
        const array<double,dim>& arr2{*rhs.data};
        #pragma omp parallel for simd reduction(+:ret) if(dim>100000)
        for(int i=0;i<dim;++i){
            ret+=arr1[i]*arr2[i];
        }
        return ret;
    }
    constexpr double abs() const{
        double ret{};
        const array<double,dim>& data{*Matrix<dim,1>::data};
        #pragma omp parallel for simd reduction(+:ret) if(dim>100000)
        for(int i=0;i<dim;++i){
            ret+=data[i]*data[i];
        }
        return std::sqrt(ret);
    }
    static constexpr Vector zero(){
        array<double,dim>* data{};
        for(int i=0;i<dim;++i){
            (*data)[i]=0;
        }
        return Vector{data};
    }
};
template<size_t size>
class SquareMatrix : public Matrix<size,size>{
public:
    SquareMatrix(array<double,size*size>* data) : Matrix<size,size>{data}{}
    SquareMatrix() : Matrix<size,size>{}{}
    SquareMatrix(const SquareMatrix& other) : Matrix<size,size>{other}{}
    SquareMatrix(const Matrix<size,size>& other) : Matrix<size,size>{other}{}
    SquareMatrix(const array<double,size*size> data) : Matrix<size,size>{data}{}
    constexpr SquareMatrix& transposeInPlace(){   
        #pragma omp parallel for if(size*size>2000)
        for(int i=0;i<size;++i){
            for(int j=i;j<size;++j){
                std::swap((*this->data)[j*size+i], (*this->data)[i*size+j]);
            }
        }
        return *this;
    }
    constexpr double determinant() const {
        return SquareMatrix{*this}.determinantInPlace();
    }
    constexpr double determinantInPlace(){
        array<double, size*size>& data{*this->data};
        double sign{1};
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
                const double factor{data[jOffset+i]/data[iOffset+i]};
                for(int k=i;k<size;++k){
                    data[jOffset+k]-=factor * data[iOffset+k];
                }
            }
        }
        double ret{1};
        for(int i=0;i<size;++i){
            ret*=data[i*size+i];
        }
        return sign * ret;
    }
    constexpr bool isOrthogonal() const{
        return (SquareMatrix{*this}.transposeInPlace()*(*this)).isIdentity();
    }
    constexpr bool isIdentity() const{
        for(int i=0;i<Matrix<size,size>::data->size();++i){
            if(i%size==i/size){
                if(Matrix<size,size>::data->at(i)!=1) return false;
            }
            else if(Matrix<size,size>::data->at(i)) return false;
        }
        return true;
    }
    constexpr bool isSymmetric() const{
        for(int i=0;i<size;++i){
            for(int j=0;j<size-i;++i){
                if(Matrix<size,size>::data->at(i*size+j)!=Matrix<size,size>::data->at(j*size+i)) return false;
            }
        }
        return true;
    }
    constexpr bool isAntiSymmetric() const{
        for(int i=0;i<size;++i){
            for(int j=0;j<size-i;++i){
                if(Matrix<size,size>::data->at(i*size+j)!=-Matrix<size,size>::data->at(j*size+i)) return false;
            }
        }
        return true;
    }
    constexpr SquareMatrix exponential() const{
        const size_t k{1ULL<<31};
        SquareMatrix m{SquareMatrix::identity()+(1.0/k)*(*this)};
        for(size_t i=1;i<k;i*=2){
            m*=m;
        }
        return m;
    }
    constexpr Vector<size> solve(Vector<size>& v) const{
        //solve with gauß algorithm
        SquareMatrix<size> m{*this};
        array<double, size*size>& data{*m.data};
        for(int i=0;i<size;++i){
            const size_t iOffset{i*size};
            if(data[iOffset+i]==0){
                bool found{false};
                for(int j=i+1;j<size;++j){
                    if(data[j*size+i]!=0){
                        m.swapRows(i,j);
                        found=true;
                        break;
                    }
                }
                if(!found){
                    return Vector<size>::zero();
                }
            }
            for(int j=i+1;j<size;++j){
                const size_t jOffset{j*size};
                const double factor{data[jOffset+i]/data[iOffset+i]};
                for(int k=i;k<size;++k){
                    data[jOffset+k]-=factor * data[iOffset+k];
                }
                (*v.data)[j]-=factor * (*v.data)[i];
            }
        }
        Vector<size> ret{};
        for(int i=0;i<size;++i){
            const size_t offset{(size-1-i)*size+size -1};
            double rhs{(*v.data)[size-1-i]};
            for(int j=0;j<i;++j){
                rhs-=ret[size-1-j]*data[offset -j];
            }
            ret[size-1-i]=rhs/data[offset -i];
        }
        return ret;
    }
    constexpr vector<Vector<size>> solve(vector<Vector<size>>& vectors){
        SquareMatrix<size> m{*this};
        array<double, size*size>& data{*m.data};
        for(int i=0;i<size;++i){
            const size_t iOffset{i*size};
            if(data[iOffset+i]==0){
                bool found{false};
                for(int j=i+1;j<size;++j){
                    if(data[j*size+i]!=0){
                        m.swapRows(i,j);
                        for(auto& v: vectors){
                            v.swapRows(i,j);
                        }
                        found=true;
                        break;
                    }
                }
                if(!found){
                    return vector<Vector<size>>{Vector<size>::zero()};
                }
            }
            for(int j=i+1;j<size;++j){
                const size_t jOffset{j*size};
                const double factor{data[jOffset+i]/data[iOffset+i]};
                data[jOffset+i]=factor;
                for(int k=i+1;k<size;++k){
                    data[jOffset+k]-=factor * data[iOffset+k];
                }
                (*vectors[0].data)[j]-=factor * (*vectors[0].data)[i];
            }
        }
        for(int i=0;i<size;++i){
            const size_t offset{(size-1-i)*size+size -1};
            double rhs{vectors[0][size-1-i]};
            for(int j=0;j<i;++j){
                rhs-=vectors[0][size-1-j]*data[offset -j];
            }
            vectors[0][size-1-i]=rhs/data[offset -i];
        }
        for(int v=1;v<vectors.size();++v){
            for(int i=0;i<size;++i){
                const size_t offset{i*size};
                double rhs{vectors[v][i]};
                for(int j=0;j<i;++j){
                    rhs-=vectors[v][j]*data[offset + j];
                }
                vectors[v][i]=rhs;
            }
            for(int i=0;i<size;++i){
                const size_t offset{(size-1-i)*size+size -1};
                double rhs{vectors[v][size-1-i]};
                for(int j=0;j<i;++j){
                    rhs-=vectors[v][size-1-j]*data[offset -j];
                }
                vectors[v][size-1-i]=rhs/data[offset -i];
            }
        }
        return vectors;
    }
    static SquareMatrix filledWith(const double value){
        SquareMatrix ret{};
        std::fill(ret.data->begin(), ret.data->end(), value);
        return ret;
    }
    static SquareMatrix identity(){
        SquareMatrix ret{};
        for(int i=0;i<size;++i) {
            (*ret.data)[i*size+i]=1;
        }
        return ret;
    }
    template<size_t>
    friend constexpr SquareMatrix operator*(const SquareMatrix& m1, const SquareMatrix& m2);
    constexpr SquareMatrix& operator*=(const SquareMatrix& m) {
        array<double, size*size>* ret{new array<double, size*size>{}};
        array<double, size*size>* tmp{new array<double, size*size>{}};
        array<double, size*size>& arr1{*m.data};
        array<double, size*size>& arr2{*tmp};
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
                double entry{0};
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
    constexpr SquareMatrix& operator*=(const double factor){
        #pragma omp parallel for if(size*size>=1000000)
        for(int i=0;i<this->data->size();++i){
            (*this->data)[i]*=factor;
        }
        return *this;
    }
};
template<size_t size>
constexpr SquareMatrix<size> operator*(const SquareMatrix<size>& m1, const SquareMatrix<size>& m2) {
    array<double, size*size>* ret{new array<double, size*size>{}};
    array<double, size*size>* tmp{new array<double, size*size>{}};
    array<double, size*size>& arr1{*m1.data};
    array<double, size*size>& arr2{*tmp};
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
            double entry{0};
            const size_t jOffset{j*size};
            #pragma omp simd reduction(+:entry)
            for(int k=0;k<size;++k){
                entry+=arr1[iOffset + k] * arr2[jOffset+k];
            }
            (*ret)[i*size + j] = entry;
        }
    }
    delete tmp;
    return SquareMatrix<size>{ret};
}
template<typename T>
constexpr T operator+(const T& m1, const T& m2) { return T{m1}+=m2; };
template<typename T>
constexpr T operator-(const T& m1, const T& m2) { return T{m1}-=m2; };
template<typename T>
constexpr T operator*(const T& m, const double factor) { return T{m}*=factor; }
template<typename T>
constexpr T operator*(const double factor, const T& m) { return m * factor; }
    }
}