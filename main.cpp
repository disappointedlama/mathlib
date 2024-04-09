#include"matrix.h"
int main(){
    using namespace mathlib::matrix;
    double start, end, duration;
    /*
    */
    SquareMatrix<4000> very_large_m{SquareMatrix<4000>::getRandom(0,100)};
    start = omp_get_wtime();
    very_large_m = 2 * very_large_m * very_large_m - 2 * very_large_m;
    cout<<"done after "<<omp_get_wtime()-start<<"s\n";
    start = omp_get_wtime();
    very_large_m.transpose();
    cout<<"done after "<<omp_get_wtime()-start<<"s\n";
    SquareMatrix<4> sqm{{2,1,1,0,4,3,3,1,8,7,9,5,6,7,9,8}};
    cout<<sqm.determinant()<<endl;
    cout<<sqm<<endl;
    SquareMatrix<2> sqm2{{4,3,6,3}};
    cout<<sqm2.determinant()<<endl;
    cout<<sqm2<<endl;
    SquareMatrix<2000> large_sqm{SquareMatrix<2000>::getRandom(-0.01,0.01)};
    start=omp_get_wtime();
    cout<<large_sqm.determinant()<<endl;
    cout<<"done after "<<omp_get_wtime()-start<<"s"<<endl;
    SquareMatrix<100> sqm3{SquareMatrix<100>::getRandom(-1,1)};
    Vector<100> tmp{Vector<100>::getRandom(-1,1)};
    cout<<sqm3.solve(tmp);
    SquareMatrix<4096> benchmarkMatrix{SquareMatrix<4096>::getRandom(-1,1)};
    start=omp_get_wtime();
    SquareMatrix<4096> tmp2{benchmarkMatrix*benchmarkMatrix};
    duration=omp_get_wtime()-start;
    cout<<"done after "<<duration<<"s"<<endl;
    cout<<2.0*4096*4096*4096/duration<<" flops/s"<<endl;
    start=omp_get_wtime();
    double det = benchmarkMatrix.determinantInPlace();
    duration=omp_get_wtime()-start;
    cout<<"determinant calculated after "<<duration<<"s"<<endl;
    SquareMatrix<3> m{{1,2,3,0,5,6,7,8,9}};
    cout<<m<<endl;
    vector<Vector<3>> v{{Vector<3>{{1,2,3}},Vector<3>{{2,4,6}},Vector<3>{{4,8,12}}}};
    vector<Vector<3>> v2{m.solve(v)};
    for(const auto& vec:v2){
        cout<<vec<<endl;
    }
    SquareMatrix<4> orthogonal{{0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0}};
    cout<<orthogonal.isOrthogonal()<<endl;
    cout<<orthogonal.isSymmetric()<<endl;
    cout<<SquareMatrix<4>::identity().isSymmetric()<<endl;
    cout<<SquareMatrix<2>{{1,4,1,1}}.exponential()<<endl;
    SquareMatrix<3> matr{{1,2,3,4,5,6,7,8,9}};
    matr*=2;
    return 0;
}