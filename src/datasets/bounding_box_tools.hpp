#ifndef __BOUNDING_BOX_TOOLS_HPP__
#     define __BOUNDING_BOX_TOOLS_HPP__

#include <string>
#include <vector>
#include <cuv/basics/tensor.hpp>

namespace cimg_library{
    template<class T>
        class CImg;
};

namespace cuvnet
{
    /** 
     * Tools for dealing with (sub-) images and bounding boxes.
     * @ingroup datasets
     */
    namespace bbtools
    {

        struct rectangle{
            /// a 4-tuple of box coordinates
            /// @{
            int xmin;
            int xmax;
            int ymin;
            int ymax;
            /// @}
        };


        struct object {
            unsigned int klass;  ///< the index of the class this object belongs to
            bool truncated;      ///< true if the truncated property was set in the XML file
            rectangle bb;
        };

        struct image_meta_info{
            std::string filename;   ///< image file name
            std::vector<object> objects; ///< descriptions of depicted objects

            /// coordinates of processed image in original image.
            /// The coordinates may be outside the processed image, which means 
            /// that the image is only a view on the original image.
            rectangle position_in_orig;

            /// some variables recording where the image is from
            /// @{
            /// the total number of scales at which original image is processed
            unsigned int n_scales;
            /// the total number of (sub-) images at current scale
            unsigned int n_subimages;
            /// a running number indicating at which scale of the original image we're processing
            unsigned int scale_id;
            /// a running number of images at this scale (relevant when large image is split up for processing)
            unsigned int subimage_id;
            /// @}
        };

        struct image{
            image_meta_info meta;
            cimg_library::CImg<unsigned char>* porig;

            image(){}
            image(const std::string& fn);
            ~image();

            void transpose();

            private:
            image(const image&);
            image& operator=(const image&);
        };

        struct object_filter;

        struct sub_image{
            const image& original_img;
            cimg_library::CImg<unsigned char>* pcut;
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
            object object_relative_to_subimg(const object& o);

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
            sub_image& mark_objects(int type=0, unsigned char color=255, float scale=1.f, cimg_library::CImg<unsigned char>* img1=NULL);

            /**
             * scale a cropped sub_image such that larger dimension is a fixed given size.
             */
            sub_image& scale_larger_dim(int size);

            /**
             * the margin outside the original image is filled with the supplied color.
             * @param color the fill-color
             * @param img if not given, will paint into the pcut variable.
             * @throw runtime_error if neither img nor pcut available.
             */
            sub_image& fill_padding(int color, cimg_library::CImg<unsigned char>* img = NULL);

            /**
             * cuts padding from an image, resulting only in parts that have a counterpart in the original image.
             * @param img if not given, works on internal pcut image
             * @throw runtime_error if neither img nor pcut available.
             */
            sub_image& remove_padding(cimg_library::CImg<unsigned char>* img = NULL);

            /**
             * inquire whether there are objects inside this subimage
             */
            bool has_objects();

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
         * @param si the sub-image the object [might] be in
         * @param o  the object we're considering
         */
        struct object_filter{
            virtual bool filter(const sub_image& si, const object& o);
        };

        /**
         * only allow objects of a single class, in addition to
         * the requirements by object_filter.
         */
        struct single_class_object_filter
            : public object_filter
        {
            unsigned int klass;
            single_class_object_filter(unsigned int k):klass(k){}
            virtual bool filter(const sub_image& si, const object& o);
        };

        /**
         * filter objects which have roughly the same scale, in addition to
         * the requirements by object_filter.
         */
        struct similar_scale_object_filter
            : public object_filter
        {
            float min_frac, max_frac;
            similar_scale_object_filter(float minf, float maxf):min_frac(minf), max_frac(maxf){}
            virtual bool filter(const sub_image& si, const object& o);
        };

        /**
         * join two filters by conjunction
         */
        template<class A, class B>
        struct and_object_filter
        : public object_filter
        {
            object_filter& a;
            object_filter& b;
            and_object_filter(object_filter& _a, object_filter& _b):a(_a), b(_b){}
            virtual bool filter(const sub_image& si, const object& o){
                return a.filter(si,o) && b.filter(si,o);
            }
        };
        

    }

}
#endif /* __BOUNDING_BOX_TOOLS_HPP__ */