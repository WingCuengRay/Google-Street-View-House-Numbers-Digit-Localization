#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include<string.h>
#include<fstream>
#include<iostream>
#include<algorithm>
using namespace cv;
using namespace std;

ofstream out;

struct sort_condition
{
    bool operator()(const Rect &i,const Rect &j)
    {
        //sorting based on their center x value
        return ((i.x+i.width/2)<(j.x+j.width/2));
        //return (i.x<j.x);

    }
};

class detect
{
public:

    CascadeClassifier digit_cascade[10];
    fstream out;
    string base, address;
    vector<Rect> digits, all_digits, combined, filtered;
    vector<float> area;
    Mat frame_gray,result, result2, gray, img[3], temp_copy;
    float ratios[5];
    Scalar colors[3];
    int curr_x, curr_y, curr_width, curr_height, imNo;
    Rect ans;

    vector <float> confidence;

    //　构造阶段主要读取了 10 个 classifier 的 cascade 参数
    detect(String add="/home/ray/Code/Google-Street-View-House-Numbers-Digit-Localization/cascades/cascade")
    {
        ratios[0] = 1;
        ratios[1] = 2;
        ratios[2] = 1.25;
        ratios[3] = 1.5;
        ratios[4] = 1.75;

        colors[0] = (0,0,255);
        colors[1] = (0,255,0);
        colors[2] = (255,0,0);

        base = add;
        imNo=0;
        out.open("newCPU.txt", ios::out);
        load();
    }

    void load()
    {
        for(int i=0; i<10; i++)
        {
            ostringstream ss;
            ss<<i;
            address = base + ss.str() + "/cascade.xml";
            if( !digit_cascade[i].load(address))
            {
                printf("--(!)Error loading cascade number : %d\n",i);
            }

        }
    }

    pair<float, float> stats(vector<float> &area)
    {
        float sum=0.0, sumsq=0.0;
        int len = area.size();
        for(int i=0; i<len; i++)
        {
            sum+=area[i];
            sumsq+=area[i]*area[i];
        }
        pair <float, float> musigma;
        musigma.first = sum/len;        // mu:　均值
        musigma.second = sqrt((sumsq/len)-(musigma.first*musigma.first));   // sigma: 类方差，与方差公式不同
        //cout << musigma.first << ' ' << musigma.second << endl;
        return musigma;
    }

    /**
    ** @func: 使用正态分布将处于边缘的 height*width 的区域过滤掉
    ** @param:
    **      all_digits -- vector<Rect>，保存了所有检测到的数字区域的坐标（包括 false positive）
    **      filtered -- vector<Rct>，空 vector，用于返回过滤后的区域坐标
    **      dist -- float，相当于正态分布的一个概率阈值
    **
    **/
    void areafilter(vector<Rect> &all_digits, vector<Rect> &filtered, float dist = 0.75)
    {
        area.clear();
        for(int i=0; i<all_digits.size(); i++)
        {
            area.push_back(all_digits[i].height*all_digits[i].width);
        }

        //get the mu and sugma value of the areas
        pair<float, float> musigma = stats(area);

        for(int i=0; i<all_digits.size(); i++)
        {
            //discard the bboxs thats don't lie within dist*sigma of mean
            // 根据 height * width 的乘积舍弃了一些处于正态分布边缘的 area
            if(abs(musigma.first-(all_digits[i].height*all_digits[i].width)) <= (dist*musigma.second+25))
            {
                filtered.push_back(all_digits[i]);
            }
        }
    }


    /**
    ** @func: 将所有图片按照重叠程度合并（聚簇）
    ** @param: 
    **      all_digits -- vector<Rect>,这里的命名有误导嫌疑，这个 vector 存放的区域是经过 filter 的区域，而不是 all_digits
    **      combined   -- vector<Rect>,　合并完后的矩阵坐标
    ** @pre-condition: all_digits yi an zhao juzhen zhongxin chongxiao daoda paixu
    **/
    void cluster(vector<Rect> &all_digits, vector<Rect> &combined)
    {
        // cc represent the count of overlapped rectangle in one cluster
        float cc = 1;
        Rect overlap, temp(all_digits[0]);
        for(int i=1; i<all_digits.size(); i++)
        {
            //The intersection of two area
            overlap = all_digits[i] & temp ;
            //clustered rectangle size = mean of all bbox in the cluster
            // area() == width*height
            if( overlap.area() > 0.5*temp.area() || overlap.area() > 0.5*all_digits[i].area())
            {
                //calculation of running mean
                //take the central point of all overlaped rectangle across x and y axis
                temp = Rect((temp.tl()*cc + all_digits[i].tl())*(1/(cc+1)),
                            (temp.br()*cc + all_digits[i].br())*(1/(cc+1)));
                cc++;
            }
            else
            {
                //no more rectangles can be added to cluster, save temp in combined.
                combined.push_back(temp);
                temp = all_digits[i];
                cout<<"Rectangles clustered:"<<cc<<"\n";
                confidence.push_back(cc);
                cc=1;
            }
        }
        combined.push_back(temp);//pushing the last cluster
        cout<<"Rectangles clustered:"<<cc<<"\n";
        confidence.push_back(cc);
    }

    void eval( Mat image, int enlarge=1)
    {
        result = image.clone();
        result2 = image.clone();
        temp_copy = image.clone();
        cvtColor( image, image, CV_BGR2GRAY );

        // ratio[] 定义了 multi-scale 中不同的缩放比例，这里只使用了两个缩放比例：１和２
        for(int k=0; k<2; k++)
        {
            // x　轴缩放 ratios[k]*enlarge 倍，　y 轴缩放 enlarge 倍
            resize(image,img[k],Size(0,0),ratios[k]*enlarge,enlarge);


            //　将检测到的 10　个数字的区域全部保存在 all_digit 中
            for(int i=0; i<10; i++)
            {
                // digits 是由　Rect 组成的 vector
                // digits 用于返回检测到的物体的坐标
                digit_cascade[i].detectMultiScale( img[k], digits, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(20, 30), Size(400,600) );

                ostringstream ss;
                ss<<i;
                for( int j = 0; j < digits.size(); j++ )
                {
                    //将缩放后产生的坐标转换为原尺寸坐标
                    curr_x = digits[j].x/(ratios[k]*enlarge);
                    curr_y = digits[j].y/enlarge;
                    curr_width = digits[j].width/(enlarge*ratios[k]);
                    curr_height = digits[j].height/enlarge;
                    //cout << curr_x << ' ' << curr_y << ' ' << curr_width << ' ' << curr_height << endl;

                    //检测到所有区域的坐标都保存在 all_digits (vector)　中
                    all_digits.push_back(Rect(curr_x, curr_y, curr_width, curr_height));

                    //　rectangle() 函数用于画矩形
                    rectangle(result, Rect(curr_x, curr_y, curr_width, curr_height), Scalar(255,255,0), 1, 8, 0);
                    putText(result, ss.str() , Point(curr_x, curr_y+curr_height),  CV_FONT_HERSHEY_PLAIN, 1.0, (255,0,0) );
                }
                digits.clear();
            }
        }
        //　根据矩形区域的 x　轴中心坐标排序
        sort(all_digits.begin(), all_digits.end(), sort_condition());
        //cout << "all_digits length:" << all_digits.size() << endl;

        if(all_digits.size()>0)
        {
            areafilter(all_digits, filtered);
            //cout << "filtered length:" << filtered.size() << endl;
            cluster(filtered, combined);
            //cout << "combined length:" << combined.size() << endl;

            //display clustered bbox
            for(int i=0; i<combined.size(); i++)
            {
                ans = Rect(combined[i].x, combined[i].y + 0.1*combined[i].height,
                           combined[i].width, 0.8*combined[i].height );

                rectangle(result2, ans, Scalar(255), 1, 8, 0);
            }

            imNo++;
            write();

            imshow("result", result );
            imshow("combined", result2 );

        }
        else
        {
            //if no detection then call eval again with enlarged image
            cout<<"imNo: "<<imNo<<" "<<endl;
            cout<<result.rows<<" "<<result.cols<<endl;
            eval(temp_copy,2*enlarge);
        }


        all_digits.clear();
        filtered.clear();
        combined.clear();
        confidence.clear();
    }

    /**
    ** @func: 将经过 filter 和 cluster 的坐标写入文件中
    **/
    void write()
    {
        int mini = 4;
        Rect temporary;
        int tempc;      
        // 讲　digit area 按置信度从小到大排序
        for(int i=0; i<combined.size(); i++)
        { 
            for(int j=0; j<combined.size(); j++)
            {
                if(confidence[j] < confidence[i])
                {
                    temporary = combined[i];
                    combined[i] = combined[j];
                    combined[j] = temporary;

                    tempc = confidence[i];
                    confidence[i] = confidence[j];
                    confidence[j] = tempc;
                }
            }
        }
        int combinedSize=combined.size();
        out<<imNo<<" "<<std::min(combinedSize, mini);

        // mini 的值代表一张图片数字个数上限为　4
        for(int i =0; i<combined.size(); i++)
        {
            out<<" "<<combined[i].x<<" "<<combined[i].y<<" "<<combined[i].width<<" "<<combined[i].height;//<<" "<<confidence[i];
            if(i >= mini-1)
                break;
        }
        out<<"\n";

    }
    ~detect()
    {
        out.close();
    }

};

int main()
{

    namedWindow("result",2);
    namedWindow("combined",2);
    Mat image, image_truth, frame;
    detect detector;

    string address = "/home/ray/Code/machine-learning/projects/capstone/data/svhn/full/test/";
    //string address = "C:/Users/student/Desktop/adc/svhn/train/";
    string filename, cas, converted;
    for(int i =1; i<=100; i++)
    {
        cout<<i<<endl;
        ostringstream ss;
        ss<<i;
        //cout<<"address : "<<address<<endl;
        filename = address + ss.str() + ".png";

        image = imread(filename,1);
        detector.eval(image);
        /*int c = waitKey(0);
        if( (char)c == 27 )
        {
            destroyAllWindows();
            break;
        }*/

    }
    return 0;
}
