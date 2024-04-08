#include"matrix.h"
int main(){
    /*
    SquareMatrix<3> m{{1,2,3,4,5,6,7,8,9}};
    cout<<m<<endl;
    cout<<m+m<<endl;
    cout<<2*m<<endl;
    cout<<((2*m)!=(m+m))<<endl;
    m+=m;
    cout<<m<<endl;
    Vector<3> v{{1,2,3}};
    cout<<v<<endl;
    cout<<m*v<<endl;
    cout<<Matrix<3,3>::getRandom(0,100)<<endl;
    SquareMatrix<60> large_m{ SquareMatrix<60>::getRandom(0,100)};
    large_m = large_m*2;
    SquareMatrix<4000> very_large_m{SquareMatrix<4000>::getRandom(0,100)};
    double start = omp_get_wtime();
    very_large_m = 2 * very_large_m * very_large_m - 2 * very_large_m;
    cout<<"done after "<<omp_get_wtime()-start<<"s\n";
    cout<<m.swapCols(0,2)<<endl;
    cout<<m.swapRows(0,2)<<endl;
    start = omp_get_wtime();
    very_large_m.transpose();
    cout<<"done after "<<omp_get_wtime()-start<<"s\n";
    Vector<100000000> v2{Vector<100000000>::getRandom(1,1)};
    cout<<v2*v2<<endl;
    cout<<v2.abs()<<endl;
    cout<<m.transpose()<<endl;
    SquareMatrix<4> sqm{{2,1,1,0,4,3,3,1,8,7,9,5,6,7,9,8}};
    cout<<sqm.determinant()<<endl;
    cout<<sqm<<endl;
    SquareMatrix<2> sqm2{{4,3,6,3}};
    cout<<sqm2.determinant()<<endl;
    cout<<sqm2<<endl;
    SquareMatrix<2000> large_sqm{SquareMatrix<2000>::getRandom(-0.01,0.01)};
    double start=omp_get_wtime();
    cout<<large_sqm.determinant()<<endl;
    cout<<"done after "<<omp_get_wtime()-start<<"s"<<endl;
   SquareMatrix<100> sqm3{SquareMatrix<100>::getRandom(-1,1)};
   Vector<100> tmp{Vector<100>::getRandom(-1,1)};
   cout<<sqm3.solve(tmp);
   SquareMatrix<4096> benchmarkMatrix{SquareMatrix<4096>::getRandom(-1,1)};
   double start=omp_get_wtime();
   SquareMatrix<4096> tmp2{benchmarkMatrix*benchmarkMatrix};
   double duration=omp_get_wtime()-start;
   cout<<"done after "<<duration<<"s"<<endl;
   cout<<2.0*4096*4096*4096/duration<<" flops/s"<<endl;
   start=omp_get_wtime();
   double det = benchmarkMatrix.determinantInPlace();
   duration=omp_get_wtime()-start;
   cout<<"determinant calculated after "<<duration<<"s"<<endl;
    */
   SquareMatrix<3> m{{1,2,3,0,5,6,7,8,9}};
   vector<Vector<3>> v{{Vector<3>{{1,2,3}}}};
   cout<<m.solve(v)[0]<<endl;
   return 0;
}