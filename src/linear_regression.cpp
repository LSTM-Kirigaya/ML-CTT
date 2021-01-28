#include "iostream"
#include "vector"
#include "string"
#include "fstream"
#include "Eigen/Dense"
using namespace std;
using namespace Eigen;

template<class DataType> 
using list2d = vector< vector<DataType> >;



template<class DataType>
void DataLoader(const string &file_path, list2d<DataType> &v)   // 读入数据
{
    fstream fptr(file_path.c_str(), ios::in);
    if (fptr.fail())
    {
        cout << "fail to open the file!" << endl;
        exit(-1);
    }
    while (!fptr.eof())
    {
        DataType buf;
        vector<DataType> vv;
        for (int i = 0; i < 14; ++ i)  
        {
            fptr >> buf;
            vv.push_back(buf);
            fptr.ignore();
        }
        v.push_back(vv);
    }
}

// template<class DataType>
// class LinearRegression
// {
//     public:
        
// };


int main(int argc, char const *argv[])
{
    list2d<double> v;
    DataLoader("../data/boston.txt", v);

    double ratio = 0.8;
    int offline = ratio * v.size();
    
    // 申明数据的shape
    ArrayXXd train_X = ArrayXXd(offline, 13);
    ArrayXXd train_Y = ArrayXXd(offline, 1);
    ArrayXXd test_X = ArrayXXd(v.size() - offline, 13);
    ArrayXXd test_Y = ArrayXXd(v.size() - offline, 1);

    for (int i = 0 ; i < offline; ++ i)
    {
        for (int j = 0; j < 13; ++ j)
            train_X(i, j) = v[i][j];
        train_Y(i, 0) = v[i][13];
    }

    for (int i = offline; i < v.size(); ++ i)
    {
        for (int j = 0; j < 13; ++ j)
            test_X(i, j) = v[i][j];
        test_Y(i, 0) = v[i][13];
    }


    return 0;
}
