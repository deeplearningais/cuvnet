#ifndef __CUVNET_TOY_HPP__
#     define __CUVNET_TOY_HPP__
#include <fstream>
#include <iostream>
#include "dataset.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "image_queue.cpp"

namespace cuvnet
{
    /*
     * A toy dataset*
     * @ingroup datasets
     * TODO add test labels
     */
    struct toy_dataset : public dataset{
        toy_dataset(const std::string& path, const bool out_of_n_coding = true){
            
	    unsigned int size = 32;
            bool grayscale = true;
            unsigned int nclasses = 2;
	    unsigned int sizex = 15;
	    bool first = false;           

	    cuv::tensor<float, cuv::host_memory_space> traind;
	    cuv::tensor<float, cuv::host_memory_space> trainl;

            traind.resize(cuv::extents[size][sizex*sizex]);
            trainl.resize(cuv::extents[size]);

	    //set first half of labels to one, second half to zero
	    trainl[cuv::indices[cuv::index_range(0, size/2)]] = 0.0;
	    trainl[cuv::indices[cuv::index_range(size/2, size)]] = 1.0;

	    std::vector<cv::Mat> src(size);
	    for ( unsigned int i = 0; i < size; i++){
		    std::string name = path; 
		    name.append(std::to_string(i));
		    name.append(".png");

		    //std::cout << "name: " << name << std::endl;
		    //load image
		    if (grayscale)
			    src[i] = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);
		    else
			    src[i] = cv::imread(name, CV_LOAD_IMAGE_COLOR);

                    if(! src[i].data )                              // Check for invalid input
	            {
			    std::cout <<  "Could not open or find the image" << std::endl ;
		    } else {
		        //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display
		        //cv::imshow( "Display window", image );
		        //cv::waitKey(0); 
		    }
	    }
            //convert images to tensor
	    cuvnet::image_datasets::dstack_mat2tens(traind, src /* bool reverse=false*/);

	    train_data.resize(cuv::extents[size][sizex*sizex]);
	    train_data = traind.copy();	    
	    train_data.reshape(cuv::extents[size][sizex*sizex]);

            if (out_of_n_coding){
                train_labels.resize(cuv::extents[size][nclasses]);
                train_labels = 0.f;
                for (unsigned int i = 0; i < trainl.size(); ++i){
                    train_labels(i, trainl[i]) = 1.f;
                }
            } else {
                train_labels.resize(cuv::extents[size]);
                for (unsigned int i = 0; i < trainl.size(); ++i){
                    train_labels(i) = trainl[i];
                }
            }


            binary = true;
            channels = 1;
            image_size = 28;
            std::cout << "done."<<std::endl;
        }
    };


    /*
     * A toy dataset*
     * @ingroup datasets
     * TODO add test labels
     */
    struct toy_dataset2 : public dataset{
        toy_dataset2(const std::string& path, const bool out_of_n_coding = true){
            
	    unsigned int size = 32;
            bool grayscale = true;
            unsigned int nclasses = 4;
	    unsigned int sizex = 28;
	    bool first = false;           

	    cuv::tensor<float, cuv::host_memory_space> traind;
	    cuv::tensor<float, cuv::host_memory_space> trainl;

            traind.resize(cuv::extents[size][sizex*sizex]);
            trainl.resize(cuv::extents[size]);

	    //set first half of labels to one, second half to zero
	    trainl[cuv::indices[cuv::index_range(0, 8)]] = 0.0;
	    trainl[cuv::indices[cuv::index_range(8, 16)]] = 1.0;
	    trainl[cuv::indices[cuv::index_range(16, 24)]] = 2.0;
	    trainl[cuv::indices[cuv::index_range(24, 32)]] = 3.0;

	    std::vector<cv::Mat> src(size);
	    for ( unsigned int i = 0; i < size; i++){
		    std::string name = path; 
		    name.append(std::to_string(i));
		    name.append(".png");

		    //std::cout << "name: " << name << std::endl;
		    //load image
		    if (grayscale)
			    src[i] = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);
		    else
			    src[i] = cv::imread(name, CV_LOAD_IMAGE_COLOR);

                    if(! src[i].data )                              // Check for invalid input
	            {
			    std::cout <<  "Could not open or find the image" << std::endl ;
		    } else {
		       // cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display
		        //cv::imshow( "Display window", image );
		       // cv::waitKey(0); 
		    }
	    }
            //convert images to tensor
	    cuvnet::image_datasets::dstack_mat2tens(traind, src /* bool reverse=false*/);

	    train_data.resize(cuv::extents[size][sizex*sizex]);
	    train_data = traind.copy();	    
	    train_data.reshape(cuv::extents[size][sizex*sizex]);

            if (out_of_n_coding){
                train_labels.resize(cuv::extents[size][nclasses]);
                train_labels = 0.f;
                for (unsigned int i = 0; i < trainl.size(); ++i){
                    train_labels(i, trainl[i]) = 1.f;
                }
            } else {
                train_labels.resize(cuv::extents[size]);
                for (unsigned int i = 0; i < trainl.size(); ++i){
                    train_labels(i) = trainl[i];
                }
            }


            binary = true;
            channels = 1;
            image_size = 28;
            std::cout << "done."<<std::endl;
        }
    };

    
}


#endif /* __CUVNET_TOY_HPP__ */
