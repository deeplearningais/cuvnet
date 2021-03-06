#ifndef __BOUNDING_BOX_TOOLS_HPP__
#     define __BOUNDING_BOX_TOOLS_HPP__

#include <string>
#include <vector>
#include <cuv/basics/tensor.hpp>

namespace cv{
    class Mat;
};

namespace cuvnet
{
    /** 
     * Tools for dealing with (sub-) images and bounding boxes.
     * @ingroup bbtools
     */
    namespace bbtools
    {

        /** 
         * represents a rectangle using two 2D coordinates.
         * 
         * @ingroup bbtools
         */
        struct rectangle{
            /// @name a 4-tuple of box coordinates
            /// @{
            int xmin;
            int xmax;
            int ymin;
            int ymax;
            rectangle scale(float fact)const;
            /// @}
            rectangle(){};
            rectangle(int ymin_, int ymax_, int xmin_, int xmax_)
                    :xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_){}
            inline int area()const{
                return (ymax-ymin)*(xmax-xmin);
            }
        };


        /** 
         * represents an object in an image with its meta-information and a bounding box.
         * 
         * @ingroup bbtools
         */
        struct object {
            unsigned int klass;  ///< the index of the class this object belongs to
            bool truncated;      ///< true if the truncated property was set in the XML file
            rectangle bb;        ///< object coordinates
        };

        /**
         * contains information about objects in an image and tracks where the image comes from.
         * 
         * @ingroup bbtools
         */
        struct image_meta_info{
            std::string filename;   ///< image file name
            std::vector<object> objects; ///< descriptions of depicted objects
            float sampling_likelihood; ///< the likelihood for using this image, eg for class-stratified sampling.

            /// coordinates of processed image in original image.
            /// The coordinates may be outside the processed image, which means 
            /// that the image is only a view on the original image.
            rectangle position_in_orig;

            bool flipped;   ///< whether the image is flipped horizontally
            image_meta_info()
                :flipped(false){}
        };

        /**
         * represents an image we loaded from disk.
         *
         * Objects of this class cannot be copied.
         * 
         * @ingroup bbtools
         */
        struct image{
            /// meta information on this image.
            image_meta_info meta;
            /// if set, contains image values.
            boost::shared_ptr<cv::Mat> porig;

            /// default ctor (only for storing images in std::vector)
            image(){}
            /**
             * ctor.
             * @param fn the filename to load the image from.
             */
            image(const std::string& fn);
            /// dtor.
            ~image();

            /// rotate the image by 90 degrees (for testing)
            /// @deprecated
            void transpose();
            
            /// flip the image horizontally
            void flip_lr();

            private:
            /// images are not to be copied
            image(const image&);
            /// images are not to be copied
            image& operator=(const image&);
        };

        struct object_filter;

        struct coordinate_transformer{
            virtual void transform(int& x, int& y)const;
            virtual void inverse_transform(int& x, int& y)const;
        };

        /** 
         * Represents a region inside an image.
         * 
         * @ingroup bbtools
         */
        struct sub_image{
            const image& original_img; ///< the image we're in.
            boost::shared_ptr<cv::Mat> pcut; ///< if set, this caches the result of cutting the region we're representing from the original_image.
            rectangle pos; ///< the position in the original image (might be larger than original image!)
            object_filter* objfilt; ///< can be used to select only some objects (eg based on class)
            float scale_fact; ///< size in `pos' is multiplied by this
            
            /**
             * ctor. 
             * @param img the original image
             * @param r the rectangle in coordinates of the original image which this sub_image will represent
             */
            sub_image(const image& img, const rectangle& r);

            /**
             * ctor using whole original image. 
             * @param img the original image
             */
            sub_image(const image& img);

            /**
             * crop square region from image center.
             *
             * This works as proposed by Krizhevsky et al., first we select the
             * maximum-sized square center region of the current sub_image. 
             * Then we randomly select a smaller square within that rectangle,
             * the sizes of which are related by the frac parameter.
             * @param frac multiplier for larger square to get size of smaller square
             */
            sub_image& crop_random_square(float frac);

            /**
             * crop the image from the original image.
             *
             * @throw runtime_error if the sub_image would contain parts
             * outside the original image. Use \c crop_with_padding in that
             * case.
             */
            sub_image& crop();

            /**
             * show the cropped image (must be available)
             * @param name name of the window to be displayed
             */
            sub_image& show(const std::string name);

            /// keep aspect ratio and scale, but try to stay inside the
            /// original image.
            /// @param clip if true, clip at original image boundaries if extent is larger than original
            sub_image& constrain_to_orig(bool clip);

            /// extend rectangle to a square, by scaling shorter edge to both sides equally.
            /// Note that this might grow into a region where the original image is not available.
            sub_image& extend_to_square();

            /** 
             * crop with padding. 
             * @param border CIMG border handling type, 0: zero border, 1: nearest neighbor, 2: relect?
             */
            sub_image& crop_with_padding(int border=1);

            /**
             * determine the position of an object in the original image in a sub_image.
             * @param o the object to be processed
             */
            object object_relative_to_subimg(const object& o)const;

            /**
             * draw rectangles around the positions where objects are in the original image.
             *
             * Types can be: 0 rectangle, 1: filled rectangle, 2: blob
             * 
             * @param type how to mark the objects
             * @param color how to mark the object
             * @param scale a parameter of the method, determines eg size of blobs
             * @param img1 the picture, if NULL use internal cropped \c pcut variable
             */
            sub_image& mark_objects(int type=0, unsigned char color=255, float scale=1.f, 
                    boost::shared_ptr<cv::Mat> img1=boost::shared_ptr<cv::Mat>(), std::vector<std::vector<bbtools::rectangle> >* bboxes=NULL);

            /**
             * Scale object bounding boxes relative to their center, then 
             * describe them relative to the subimage and return them as a vector
             * of rectangle-vectors, one vector for every object class.
             *
             * @param n_classes the total number of classes in the dataset
             * @param scale how much to scale objects (around their center)
             */
            std::vector<std::vector<rectangle> > get_objects(int n_classes, float scale=1.f)const;

            /**
             * scale a cropped sub_image such that larger dimension is a fixed given size.
             */
            sub_image& scale_larger_dim(int size);

            /**
             * the margin outside the original image is filled with the supplied color.
             * @param color the fill-color
             * @param img if not given, will paint into the pcut variable.
             * @param ct if given, thi
             * @throw runtime_error if neither img nor pcut available.
             */
            sub_image& fill_padding(int color, cv::Mat* img = NULL, 
                    const coordinate_transformer& ct = coordinate_transformer());

            /**
             * cuts padding from an image, resulting only in parts that have a counterpart in the original image.
             * @param img if not given, works on internal pcut image
             * @param pos if not given, works on internal pos object
             * @throw runtime_error if neither img nor pcut available.
             */
            sub_image& remove_padding(boost::shared_ptr<cv::Mat> img = boost::shared_ptr<cv::Mat>(),
                    rectangle* pos = NULL);

            /**
             * inquire whether there are objects inside this subimage.
             */
            bool has_objects();

            /// dtor.
            ~sub_image();
        };

        /**
         * determine whether we care about a specific object in
         * an image or not.
         *
         * This filter returns true if the object bounding box
         * overlaps with the sub_image.
         *
         * Write your own filter by specializing this.
         *
         * 
         * @ingroup bbtools
         */
        struct object_filter{
            /**
             * @param si the sub-image the object [might] be in
             * @param o  the object we're considering
             * @return true iff the image/object passed the filter
             */
            virtual bool filter(const sub_image& si, const object& o);
        };

        /**
         * only allow objects of a single class, in addition to
         * the requirements by object_filter.
         * 
         * @ingroup bbtools
         */
        struct single_class_object_filter
            : public object_filter
        {
            unsigned int klass; ///< the class which is allowed
            /**
             * ctor.
             * @param k the class which is allowed.
             */
            single_class_object_filter(unsigned int k):klass(k){}
            /**
             * @param si the sub-image the object [might] be in
             * @param o  the object we're considering
             * @return true iff the image/object passed the filter
             */
            virtual bool filter(const sub_image& si, const object& o);
        };

        /**
         * filter objects which have roughly the same scale, in addition to
         * the requirements by object_filter.
         * 
         * @ingroup bbtools
         */
        struct similar_scale_object_filter
            : public object_filter
        {
            /// the minimal fraction of the image that is allowed
            float min_frac; 
            /// the maximal fraction of the image that is allowed
            float max_frac;
            /**
             * ctor.
             * @param minf the minimum fraction of the image that is allowed
             * @param maxf the maximum fraction of the image that is allowed
             */
            similar_scale_object_filter(float minf, float maxf):min_frac(minf), max_frac(maxf){}
            /**
             * @param si the sub-image the object [might] be in
             * @param o  the object we're considering
             * @return true iff the image/object passed the filter
             */
            virtual bool filter(const sub_image& si, const object& o);
        };

        /**
         * join two filters by conjunction.
         * 
         * @ingroup bbtools
         */
        template<class A, class B>
        struct and_object_filter
        : public object_filter
        {
            object_filter& a; ///< the first filter
            object_filter& b; ///< the second filter
            /**
             * ctor.
             * @param _a the first filter
             * @param _b the second filter
             */
            and_object_filter(object_filter& _a, object_filter& _b):a(_a), b(_b){}

            /**
             * filter a sub_image through both filters.
             * @param si the image to be filtered
             * @param o the object (possibly in si)
             */
            virtual bool filter(const sub_image& si, const object& o){
                return a.filter(si,o) && b.filter(si,o);
            }
        };
        
    }

}
#endif /* __BOUNDING_BOX_TOOLS_HPP__ */
