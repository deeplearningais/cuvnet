#include <sys/syscall.h> /* for pid_t, syscall, SYS_gettid */
#include<iostream>
#include<fstream>
#include<queue>
#include<iterator>
#include<boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>
#include <boost/format.hpp>
#include <cuv.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <log4cxx/logger.h>

#define cimg_use_jpeg
#include <CImg.h>

#include "voc_detection.hpp"

namespace{
    log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("voc_ds");
}


namespace cuvnet
{

    void square(cimg_library::CImg<unsigned char>& orig, int sq_size, voc_detection_dataset::image_meta_info& meta){
        if(std::max(orig.width(), orig.height()) < sq_size ){
            float fact = sq_size / (float)std::max(orig.width(), orig.height());
            orig.resize(fact * orig.width(), fact * orig.height(), -100, -100, 3, 2, 0.5f, 0.5f);
        }

        int orig_rows = orig.height();
        int orig_cols = orig.width();

        int new_rows = orig.height() >= orig.width()  ? sq_size : orig.height()/(float)orig.width() * sq_size;
        int new_cols = orig.width() >= orig.height() ? sq_size : orig.width()/(float)orig.height() * sq_size;

        //orig.blur(0.5f*orig_cols/(float)new_cols, 0.5f*orig_rows/(float)new_rows, 0, 1);
        //orig.blur(orig_cols/(float)new_cols, orig_rows/(float)new_rows, 0, 1);


        // downsample if the supplied image is not already square
        if(new_cols != sq_size || new_rows != sq_size)
            orig.resize(new_cols, new_rows, -100/* z */, -100 /* c */, 3 /* 3: linear interpolation */);

        // square
        orig.resize(sq_size, sq_size, -100, -100, 0 /* no interpolation */, 1 /* 0: zero border, 1: nearest neighbor, 2: relect? */, 0.5f, 0.5f);

        // new_pos = stretch 
        float stretchx = new_cols / (float)orig_cols;
        float stretchy = new_rows / (float)orig_rows;
        float offx     = 0.5f * stretchx * std::max(0, (int)orig_rows - (int)orig_cols);
        float offy     = 0.5f * stretchy * std::max(0, (int)orig_cols - (int)orig_rows);

        meta.xmin = stretchx * 0         + offx;
        meta.xmax = stretchx * orig_cols + offx;
        meta.ymin = stretchy * 0         + offy;
        meta.ymax = stretchy * orig_rows + offy;
        meta.xmax = std::min(sq_size-1, meta.xmax);
        meta.ymax = std::min(sq_size-1, meta.ymax);

        BOOST_FOREACH(voc_detection_dataset::object& o, meta.objects){
            o.xmin = stretchx * o.xmin + offx;
            o.ymin = stretchy * o.ymin + offy;
            o.xmax = stretchx * o.xmax + offx;
            o.ymax = stretchy * o.ymax + offy;

            assert(o.xmin <= o.xmax);
            assert(o.ymin <= o.ymax);
            //assert(o.xmin >= 0);
            //assert(o.ymin >= 0);
            //assert(o.xmax < sq_size);
            //assert(o.ymin < sq_size);
        }
    }

    int bb_teacher(
            cimg_library::CImg<unsigned char>& dst,
            cimg_library::CImg<unsigned char>& img,
            int sq_size,
            voc_detection_dataset::image_meta_info& meta,
            const voc_detection_dataset::output_properties* output_properties,
            unsigned int n_classes, float bbsize, int ttype /* 1: teacher, 0: ignore*/
            )
    {
        int final_size = (sq_size - output_properties->crop_h) / output_properties->scale_h;
        int final_start = output_properties->crop_h / 2;
        int final_scale = output_properties->scale_h;

        dst.resize(final_size, final_size, n_classes, 1, -1 /* -1: no interpolation, raw memory resize!*/);
        dst = (unsigned char) 0; // initialize w/ "ignore everywhere" resp. "no object"
        int found_obj = 0;

        BOOST_FOREACH(voc_detection_dataset::object& mo, meta.objects){
            

            // mo is in the reference frame of the sq_size input image. 
            // we first determine its position in the frame of the teacher.
            voc_detection_dataset::object o = mo;
            o.xmin -= final_start;
            o.xmax -= final_start;
            o.ymin -= final_start;
            o.ymax -= final_start;

            o.xmin /= final_scale;
            o.xmax /= final_scale;
            o.ymin /= final_scale;
            o.ymax /= final_scale;

            // make sure that the object lies completely within the region
            // represented by the teacher (which might be less than sq_size if
            // `valid' convolutions are used).
            if(o.xmin  < 0 || o.xmax > final_size)
                continue;
            if(o.ymin  < 0 || o.ymax > final_size)
                continue;


            float cx = 0.5 * (o.xmax + o.xmin);
            float cy = 0.5 * (o.ymax + o.ymin);

            unsigned int w = o.xmax - o.xmin;
            unsigned int h = o.ymax - o.ymin;
            //float ignore_amount = 1.f - std::max(0.f, std::min(1.f, 4.f*std::pow(std::max(h,w)/(float)sq_size - 0.5f, 2.f)));
            float ignore_amount = 1.f;

            if(w < final_size/4. || h < final_size/4.)
              continue;
            if(w > final_size/1.2 || h > final_size/1.2)
              continue;

            /*
             *if(ttype == 0){
             *    static int cnt = 0;
             *    cimg_library::CImg<unsigned char> simg(img, false);
             *    const unsigned char white[] = {255,255,255};
             *    simg.draw_rectangle(mo.xmin, mo.ymin, mo.xmax, mo.ymax, white, 1.f, ~0U);
             *    simg.save(boost::str(boost::format("obj-%d-%05d.jpg") % (pid_t) syscall(SYS_gettid)% cnt++ ).c_str());
             *}
             */

            //if(cx < 0 || cx >= final_size)
            //   continue;
            //if(cy < 0 || cy >= final_size)
            //   continue;

            //    TODO Do not overwrite 1s of previous objects with 0s of a non-correct-size object!
            //       --> use maximum?

            found_obj ++;

            if(ttype == 0) {
                // set ignore amount (should be larger than
                // teacher)
                unsigned char strength = 255u * ignore_amount;
                //dst.get_shared_plane(o.klass).draw_ellipse(cx, cy, 20, 20, 0.f, &strength);
                //dst.get_shared_plane(0).draw_ellipse(cx, cy, 20, 20, 0.f, &strength);

                dst.get_shared_plane(0).draw_rectangle(
                        cx-w/2+0.5, cy-h/2+0.5,0,0,
                        cx+w/2+0.5, cy+h/2+0.5,0,0,
                        strength);

            }
            else if(ttype == 1) {
                // set teacher region (smaller than `ignore')
                //dst.get_shared_plane(o.klass).draw_ellipse(cx, cy, 10, 10, 0.f, &color);
                //dst.get_shared_plane(0).draw_ellipse(cx, cy, 10, 10, 0.f, &color);
                dst.get_shared_plane(0).draw_rectangle(
                        cx-w*M_SQRT1_2/2 + 0.5, cy-h*M_SQRT1_2/2 + 0.5,0,0,
                        cx+w*M_SQRT1_2/2 - 0.5, cy+h*M_SQRT1_2/2 - 0.5,0,0,
                        255);
            }
        }
        return found_obj;
    }

    void ignore_margin(
            cimg_library::CImg<unsigned char>& dst,
            const voc_detection_dataset::image_meta_info& meta,
            const voc_detection_dataset::output_properties& op)
    {
        int xmin = ((int)meta.xmin - (int)op.crop_w) / (int)op.scale_w;
        int ymin = ((int)meta.ymin - (int)op.crop_h) / (int)op.scale_h;
        int xmax = ((int)meta.xmax - (int)op.crop_w) / (int)op.scale_w;
        int ymax = ((int)meta.ymax - (int)op.crop_h) / (int)op.scale_h;
        unsigned char color = 0;
        if(xmin > 0)
            dst.draw_rectangle(0,0,0,0,  xmin, dst.height()-1, dst.depth()-1, dst.spectrum()-1, color);
        if((int)xmax < dst.width()-1)
            dst.draw_rectangle(xmax+1,0,0,0,  dst.width()-1, dst.height()-1, dst.depth()-1, dst.spectrum()-1, color);
        if(ymin > 0)
            dst.draw_rectangle(0,0,0,0,  dst.width()-1, ymin-1, dst.depth()-1, dst.spectrum()-1, color);
        if((int)ymax < dst.height()-1)
            dst.draw_rectangle(0,ymax+1,0,0,  dst.width()-1, dst.height()-1, dst.depth()-1, dst.spectrum()-1, color);
    }


    /**
     * loads an image from file using metadata, processes it and places it in queue
     */
    struct voc_detection_file_loader{
        boost::mutex* mutex;
        std::queue<voc_detection_dataset::pattern>* loaded_data;
        const voc_detection_dataset::image_meta_info* meta;
        const voc_detection_dataset::output_properties* output_properties;
        voc_detection_file_loader(boost::mutex* m, std::queue<voc_detection_dataset::pattern>* dest, 
                const voc_detection_dataset::image_meta_info* _meta,
                const voc_detection_dataset::output_properties* op)
            : mutex(m)
              , loaded_data(dest)
              , meta(_meta)
              , output_properties(op)
        {
        }
        void split_up_bbs(cimg_library::CImg<unsigned char>& img, unsigned int crop_square_size){
            std::list<voc_detection_dataset::pattern> storage;
            BOOST_FOREACH(const voc_detection_dataset::object& bbox, meta->objects){
                
                float w = bbox.xmax - bbox.xmin;
                float h = bbox.ymax - bbox.ymin;
                float cx = (bbox.xmax + bbox.xmin)/2.f;
                float cy = (bbox.ymax + bbox.ymin)/2.f;

                float context_fact = 2.f + drand48() * 0.5f;
                float x_off = drand48() * w/4.f;
                float y_off = drand48() * w/4.f;
                float xmin = cx - context_fact * w/2.f + x_off;
                float xmax = cx + context_fact * w/2.f + x_off;
                float ymin = cy - context_fact * h/2.f + y_off;
                float ymax = cy + context_fact * h/2.f + y_off;


                // the xminmax, yminmax now a `larger' version of the original bbox. 
                // we now try to extend the smaller dimension so the new image
                // is more square, but does not go over the original image boundaries
                
                float new_w = xmax-xmin;
                float new_h = ymax-ymin;
                float max_new_wh = std::max(new_w, new_h);

                if(new_w > new_h){
                    // grow y-direction
                    ymin = ymin - (max_new_wh - new_h)/2.f;
                    ymax = ymin + max_new_wh; // now it is square
                }
                // if we moved against an image boundary, move the box so
                // that it coincides with the boundary
                if(ymax >= img.height())
                {
                    float d = ymax - (img.height() - 1);
                    ymax -= d;
                    ymin -= d;
                }
                if(ymin < 0){
                    float d = 0 - ymin;
                    ymax += d;
                    ymin += d;
                }
                // we might still be outside the valid range, but that will
                // be fixed later.
                new_h = ymax - ymin;

                if(new_h > new_w + 0.5){
                    // grow x-direction
                    xmin = xmin - (max_new_wh - new_w)/2.f;
                    xmax = xmin + max_new_wh; // now it is square
                }
                // if we moved against an image boundary, move the box so
                // that it coincides with the boundary
                if(xmax >= img.width())
                {
                    float d = xmax - (img.width() - 1);
                    xmax -= d;
                    xmin -= d;
                }
                if(xmin < 0){
                    float d = 0 - xmin;
                    xmax += d;
                    xmin += d;
                }
                // we might still be outside the valid range, but that will
                // be fixed later.

                xmin = std::max(xmin, 0.f);
                xmax = std::min(xmax, img.width()-1.f);
                ymin = std::max(ymin, 0.f);
                ymax = std::min(ymax, img.height()-1.f);

                cimg_library::CImg<unsigned char> simg = img.get_crop(xmin, ymin, xmax, ymax, false);
                /*
                 *{
                 *   static int cnt = 0;
                 *   img.save(boost::str(boost::format("img-%d-%05da.jpg") % (pid_t) syscall(SYS_gettid)% cnt++ ).c_str());
                 *   simg.save(boost::str(boost::format("img-%d-%05db.jpg") % (pid_t) syscall(SYS_gettid)% cnt++ ).c_str());
                 *}
                 */

                voc_detection_dataset::pattern pat;
                pat.meta_info = *meta;

                // remember where current crop came from
                pat.meta_info.orig_xmin = xmin;
                pat.meta_info.orig_ymin = ymin;
                pat.meta_info.orig_xmax = xmax;
                pat.meta_info.orig_ymax = ymax;

                // adjust object positions according to crop
                BOOST_FOREACH(voc_detection_dataset::object& o, pat.meta_info.objects){
                    o.xmin = o.xmin-xmin;
                    o.xmax = o.xmax-xmin;

                    o.ymin = o.ymin-ymin;
                    o.ymax = o.ymax-ymin;
                }

#define STORE_IMAGES_SEQUENTIALLY 1
#if STORE_IMAGES_SEQUENTIALLY
                // store images into a local storage and put them in
                // the queue as one block after the loops
                typedef std::list<voc_detection_dataset::pattern> list_type;
                typedef voc_detection_dataset::pattern arg_type;
                ensure_square_and_enqueue(
                        boost::bind(static_cast<void (list_type::*)(const arg_type&)>(&list_type::push_back), &storage, _1), 
                        simg, crop_square_size, pat, false /* no locking */);
#else
                // store images into the queue as soon as they are ready
                // TODO: for some reason, this does not seem to work --
                //       images look like they're cut out before smoothing?
                typedef std::queue<voc_detection_dataset::pattern> queue_type;
                typedef voc_detection_dataset::pattern arg_type;

                ensure_square_and_enqueue(
                        boost::bind( static_cast<void (queue_type::*)(const arg_type&)>(&queue_type::push)
                            , loaded_data, _1),
                        img, crop_square_size, pat, true /* locking */);
#endif
            }
#if STORE_IMAGES_SEQUENTIALLY
            boost::mutex::scoped_lock lock(*mutex);
            BOOST_FOREACH(voc_detection_dataset::pattern& pat, storage){
                loaded_data->push(pat);
            }
#endif
#undef STORE_IMAGES_SEQUENTIALLY
            //exit(1);

        }
        void split_up_scales(cimg_library::CImg<unsigned char>& img, unsigned int crop_square_size){
            static const unsigned int n_scales = 3;
            cimg_library::CImg<unsigned char> pyramid[n_scales];
            std::list<voc_detection_dataset::pattern> storage;

            // size of resulting (square) patches
            int min_crop_square_overlap = crop_square_size / 4;
            int max_crop_square_overlap = crop_square_size / 2;
            
            // highest level in pyramid should have longest edge at crop_square_size
            // lower layers are half the size... 
            int pyr0_size = std::ceil(crop_square_size * pow(2.f, (float) n_scales-1));
            float orig_to_pyr0 = pyr0_size / (float) std::max(img.width(), img.height());

            // create a Gaussian pyramid
            pyramid[0] = img.get_blur(1.0f * orig_to_pyr0, 1.0f * orig_to_pyr0, 0, 1);
            pyramid[0].resize(orig_to_pyr0*img.width(), orig_to_pyr0*img.height(), -100, -100, 3);
            for(unsigned int scale=1; scale < n_scales; scale++){
                pyramid[scale] = pyramid[scale-1].get_blur(1.f, 1.f, 0, 1);
                pyramid[scale].resize(pyramid[scale].width()/2, pyramid[scale].height()/2, -100, -100, 3);
            }

            // take apart images from each scale of the pyramid
            for (unsigned int scale = 0; scale < n_scales; ++scale){
                cimg_library::CImg<unsigned char>& pimg = pyramid[scale];
                int scaled_height = pimg.height();
                int scaled_width  = pimg.width();

                // determine number of subimages in each direction assuming minimum overlap
                // this will yield a large number of subimages, which we can then relax.
                int n_subimages_y = ceil((scaled_height-min_crop_square_overlap) / (float)(crop_square_size-min_crop_square_overlap));
                int n_subimages_x = ceil((scaled_width -min_crop_square_overlap) / (float)(crop_square_size-min_crop_square_overlap));

                // determine the /actual/ overlap we get, when images are evenly distributed
                int overlap_y=0, overlap_x=0;
                if(n_subimages_y > 1)
                    overlap_y = (n_subimages_y*crop_square_size - scaled_height) / (n_subimages_y-1);
                if(n_subimages_x > 1)
                    overlap_x = (n_subimages_x*crop_square_size - scaled_width) / (n_subimages_x-1);

                // adjust number of images if overlap not good
                while(n_subimages_y > 1  &&  overlap_y > max_crop_square_overlap){
                    n_subimages_y --;
                    if(n_subimages_y == 1)
                        overlap_y = 0;
                    else
                        overlap_y = (n_subimages_y*crop_square_size - scaled_height) / (n_subimages_y-1);
                }
                while(n_subimages_x > 1  &&  overlap_x > max_crop_square_overlap){
                    n_subimages_x --;
                    if(n_subimages_x == 1)
                        overlap_x = 0;
                    else
                        overlap_x = (n_subimages_x*crop_square_size - scaled_width) / (n_subimages_x-1);
                }

                // determine the /actual/ stride we need to move at
                int stride_y = crop_square_size;
                int stride_x = crop_square_size;

                for (int sy = 0; sy < n_subimages_y; ++sy)
                {
                    // determine starting point of square in y-direction
                    int ymin = std::max(0, sy * stride_y - sy * overlap_y);
                    int ymax = ymin + crop_square_size;

                    // move towards top if growing over bottom
                    if(ymax >= pimg.height()){
                        int diff = ymax - pimg.height() - 1;
                        ymax  = std::max(0, ymax - diff);
                        ymin  = std::max(0, ymin - diff);
                        if(ymin == 0) // we hit both edges: use whole range
                            ymax = pimg.height()-1;
                    }

                    for (int sx = 0; sx < n_subimages_x; ++sx)
                    {
                        // determine starting point of square in x-direction
                        int xmin = std::max(0, sx * stride_x - sx * overlap_x);
                        int xmax = xmin + crop_square_size;

                        // move towards top if growing over bottom
                        if(xmax >= pimg.width()){
                            int diff = xmax - pimg.width() - 1;
                            xmax  = std::max(0, xmax - diff);
                            xmin  = std::max(0, xmin - diff);
                            if(xmin == 0) // we hit both edges: use whole range
                                xmax = pimg.width()-1;
                        }

                        cimg_library::CImg<unsigned char> simg = pimg.get_crop(xmin, ymin, xmax, ymax, false);

                        voc_detection_dataset::pattern pat;
                        pat.meta_info = *meta;

                        float scale_x = img.width() / (float)scaled_width;
                        float scale_y = img.height() / (float)scaled_height;

                        // remember where current crop came from
                        pat.meta_info.orig_xmin = xmin * scale_x;
                        pat.meta_info.orig_ymin = ymin * scale_y;
                        pat.meta_info.orig_xmax = xmax * scale_x;
                        pat.meta_info.orig_ymax = ymax * scale_y;

                        pat.meta_info.n_scales    = n_scales;
                        pat.meta_info.n_subimages = n_subimages_x * n_subimages_y;
                        pat.meta_info.scale_id    = scale;
                        pat.meta_info.subimage_id = sy * n_subimages_x + sx;

                        // adjust object positions according to crop
                        BOOST_FOREACH(voc_detection_dataset::object& o, pat.meta_info.objects){
                            o.xmin = o.xmin/scale_x - xmin;
                            o.xmax = o.xmax/scale_x - xmin;

                            o.ymin = o.ymin/scale_y - ymin;
                            o.ymax = o.ymax/scale_y - ymin;
                        }

#define STORE_IMAGES_SEQUENTIALLY 1
#if STORE_IMAGES_SEQUENTIALLY
                        // store images into a local storage and put them in
                        // the queue as one block after the loops
                        typedef std::list<voc_detection_dataset::pattern> list_type;
                        typedef voc_detection_dataset::pattern arg_type;
                        ensure_square_and_enqueue(
                                boost::bind(static_cast<void (list_type::*)(const arg_type&)>(&list_type::push_back), &storage, _1), 
                                simg, crop_square_size, pat, false /* no locking */);

                        if(0){
                            voc_detection_dataset::pattern pat = storage.back();
                            cuv::tensor<float, cuv::host_memory_space> tmp = pat.img.copy();
                            tmp -= cuv::minimum(tmp);
                            tmp /= cuv::maximum(tmp) + 0.0001f;
                            tmp *= 255.f;
                            cuv::libs::cimg::save(tmp, boost::str(boost::format("image-s%02d-c%05d.jpg") % pat.meta_info.scale_id % pat.meta_info.subimage_id ));
                        }
#else
                        // store images into the queue as soon as they are ready
                        // TODO: for some reason, this does not seem to work --
                        //       images look like they're cut out before smoothing?
                        typedef std::queue<voc_detection_dataset::pattern> queue_type;
                        typedef voc_detection_dataset::pattern arg_type;

                        ensure_square_and_enqueue(
                                boost::bind( static_cast<void (queue_type::*)(const arg_type&)>(&queue_type::push)
                                    , loaded_data, _1),
                                img, crop_square_size, pat, true /* locking */);
#endif
                    }
                }
            }
#if STORE_IMAGES_SEQUENTIALLY
            boost::mutex::scoped_lock lock(*mutex);
            BOOST_FOREACH(voc_detection_dataset::pattern& pat, storage){
                loaded_data->push(pat);
            }
#endif
            //exit(1);

        }
        template<class T>
        void ensure_square_and_enqueue(T output, cimg_library::CImg<unsigned char>& img, unsigned int sq_size, voc_detection_dataset::pattern& pat, bool lock){
            cimg_library::CImg<unsigned char> tch;
            cimg_library::CImg<unsigned char> ign;
            unsigned int n_classes = 1;
            square(img, sq_size, pat.meta_info);    // ensure image size is sq_size x sq_size
            bool found_obj =
            bb_teacher(tch, img, sq_size, pat.meta_info, output_properties, n_classes, .6, 1); // generate teacher for image (ellipse)
            if(!found_obj)
                return;
            bb_teacher(ign, img, sq_size, pat.meta_info, output_properties, n_classes, 1., 0); // generate teacher for image (rect)

            ignore_margin(ign, pat.meta_info, *output_properties);
            //tch = 255 - (ign - tch).cut(0,255);
            tch = tch.cut(0,255);
            //ign = (ign - tch).cut(0,255);

            //ign.blur(5.f,5.f,0.f);
            //tch.blur(5.f);


            // convert to cuv
            //img.RGBtoYCbCr();
            pat.img.resize(cuv::extents[3][sq_size][sq_size]);
            pat.tch.resize(cuv::extents[tch.depth()][tch.height()][tch.width()]);
            pat.ign.resize(cuv::extents[ign.depth()][ign.height()][ign.width()]);
            cuv::convert(pat.img, cuv::tensor<unsigned char,cuv::host_memory_space>(cuv::extents[3][sq_size][sq_size] , img.data()));
            cuv::convert(pat.tch, cuv::tensor<unsigned char,cuv::host_memory_space>(cuv::extents[tch.depth()][tch.height()][tch.width()], tch.data()));
            cuv::convert(pat.ign, cuv::tensor<unsigned char,cuv::host_memory_space>(cuv::extents[ign.depth()][ign.height()][ign.width()], ign.data()));

            pat.img /= 255.f;
            pat.img -= 0.5f;
            pat.ign /= 255.f;

            // set ignore to a non-zero value where teacher is "no object".
            // this reweights errors, so that it is easier to produce a
            // "positive" output.
            //cuv::tensor<unsigned char, cuv::host_memory_space> idx(pat.tch.shape());
            //cuv::apply_scalar_functor(idx, pat.tch, cuv::SF_LT, 176);
            //cuv::apply_scalar_functor(pat.ign, cuv::SF_MIN, 0.001f, &idx);

            // set teacher to "0" where ignore=1 (for visualization)
            cuv::tensor<unsigned char, cuv::host_memory_space> idx(pat.tch.shape());
            cuv::apply_scalar_functor(idx, pat.ign, cuv::SF_LT, 0.05f);
            cuv::apply_scalar_functor(pat.tch, cuv::SF_MULT, 0.0f, &idx);

#if 1
            pat.tch /= 127.f;
            pat.tch -=   1.f;
#else
            pat.tch /= 255.f;
#endif

            // put in pipe
            if(lock) {
                boost::mutex::scoped_lock lock(*mutex);
                //loaded_data->push(pat);
                output(pat);
            }else
                output(pat);
        }


        void operator()(){
            cimg_library::CImg<unsigned char> img;
            img.load_jpeg(meta->filename.c_str()); // load image from file

            if(0){
                // we do not want to use sliding window, and only the scale of
                // the image itself.
                voc_detection_dataset::pattern pat;
                pat.meta_info = *meta;

                // a single crop!
                pat.meta_info.orig_xmin = 0;
                pat.meta_info.orig_ymin = 0;
                pat.meta_info.orig_xmax = img.width()-1;
                pat.meta_info.orig_ymax = img.height()-1;
                pat.meta_info.n_scales    = 1;
                pat.meta_info.n_subimages = 1;
                pat.meta_info.scale_id    = 0;
                pat.meta_info.subimage_id = 0;

                typedef std::queue<voc_detection_dataset::pattern> queue_type;
                typedef voc_detection_dataset::pattern arg_type;
                
                ensure_square_and_enqueue(
                        boost::bind( static_cast<void (queue_type::*)(const arg_type&)>(&queue_type::push)
                            , loaded_data, _1),
                        img, 176, pat, true);
            }else{
                // split up the image into multiple scales, extract images with
                // size 128x128 on all scales, and enqueue them.
                //split_up_scales(img, 176);
                split_up_bbs(img, 176);
            }
        }
    };
    
    /**
     * loads data from disk using meta-data (asynchronously)
     */
    class voc_detection_pipe{
        private:
            mutable boost::mutex m_loaded_data_mutex; /// workers use this to ensure thread-safe access to m_loaded_data.

            const std::vector<voc_detection_dataset::image_meta_info>& dataset;
            const voc_detection_dataset::output_properties& m_output_properties;

            std::queue<voc_detection_dataset::pattern> m_loaded_data;
            
            unsigned int m_min_pipe_len; ///< if pipe is shorter than this, start threads to make it grow
            unsigned int m_max_pipe_len; ///< if pipe is longer than this, stop filling it
            unsigned int m_n_threads;    ///< number of worker threads to be spawned
            bool m_request_stop; ///< if true, main loop stops
            bool m_running; ///< true between start() and request_stop()

            boost::thread m_thread; ///< can be used to stop pipe externally

        public:

            /**
             * constructor.
             *
             * @param ds dataset metadata
             * @param n_threads if 0, use number of CPUs
             * @param min_pipe_len when to start filling pipe
             * @param max_pipe_len when to stop filling pipe
             */
            voc_detection_pipe(
                    const std::vector<voc_detection_dataset::image_meta_info>& ds,
                    const voc_detection_dataset::output_properties& op,
                    unsigned int n_threads=0, unsigned int min_pipe_len=32, unsigned int max_pipe_len=0)
                : dataset(ds)
                , m_output_properties(op)
                , m_min_pipe_len(min_pipe_len)
                , m_max_pipe_len(max_pipe_len > min_pipe_len ? max_pipe_len : 3 * min_pipe_len)
                , m_n_threads(n_threads)
                , m_request_stop(false)
                , m_running(false)
            {
                if(m_n_threads == 0)
                   m_n_threads = boost::thread::hardware_concurrency();
                //m_n_threads = 1;
            }

            /**
             * destructor, stops thread.
             */
            ~voc_detection_pipe(){
                if(m_running)
                    request_stop();
            }

            /**
             * call this externally to stop pipe thread
             */
            void request_stop(){ 
                m_request_stop = true; 
                m_thread.join();
                m_running = false;
            }

            void get_batch(std::list<voc_detection_dataset::pattern>& dest, unsigned int n){
                if(m_min_pipe_len < n){
                    m_min_pipe_len = n;
                    m_max_pipe_len = 3 * n;
                }
                // TODO: use boost condition_variable!
                while(size() < n)
                    boost::this_thread::sleep(boost::posix_time::millisec(10));

                // TODO: when getting lock fails, loop again above!
                //       that way, multiple clients can use the same queue
                boost::mutex::scoped_lock lock(m_loaded_data_mutex);
                for (unsigned int i = 0; i < n; ++i) {
                    dest.push_back(m_loaded_data.front());
                    m_loaded_data.pop();
                }
            }

            /** start filling the queue  of ready-to-use elements */
            void start(){
                m_thread =  boost::thread(boost::ref(*this));
                m_running = true;
            }

            /** return the number of ready-to-use data elements */
            unsigned int size()const{
                boost::mutex::scoped_lock lock(m_loaded_data_mutex);
                return m_loaded_data.size();
            }

            
            /** main loop, ensures queue stays filled and stops when \c m_request_stop is true */
            void operator()(){
                unsigned int cnt = 0;
                std::vector<unsigned int> idxs(dataset.size());
                for(unsigned int i=0;i<dataset.size();i++)
                    idxs[i] = i;

                while(!m_request_stop){

                    unsigned int size = this->size();

                    if(size < m_min_pipe_len){
                        boost::thread_group grp;
                        for (unsigned int i = 0; i < m_n_threads && i+size < m_max_pipe_len; ++i)
                        {
                            grp.create_thread(
                                    voc_detection_file_loader(
                                        &m_loaded_data_mutex,
                                        &m_loaded_data,
                                        &dataset[idxs[cnt]],
                                        &m_output_properties));
                            cnt = (cnt+1) % dataset.size();
                            if(cnt == 0) {
                                LOG4CXX_INFO(g_log, "Roundtrip through dataset completed. Shuffling.");
                                std::random_shuffle(idxs.begin(), idxs.end());
                            }
                        }
                        grp.join_all();
                    }
                    boost::this_thread::sleep(boost::posix_time::millisec(10));
                }
            }
    };

    void voc_detection_dataset::set_output_properties(
            unsigned int scale_h, unsigned int scale_w,
            unsigned int crop_h, unsigned int crop_w){
        m_output_properties.scale_h = scale_h;
        m_output_properties.scale_w = scale_w;
        m_output_properties.crop_h = crop_h;
        m_output_properties.crop_w = crop_w;
    }
    void voc_detection_dataset::switch_dataset(voc_detection_dataset::subset ss, int n_threads){

        n_threads = n_threads == 0 ? m_n_threads : n_threads;

        if(m_pipe)
            m_pipe->request_stop();

        switch(ss){
            case SS_TRAIN:
                m_pipe.reset(new voc_detection_pipe(m_training_set, m_output_properties, n_threads));
                break;
            case SS_VAL:
                m_pipe.reset(new voc_detection_pipe(m_val_set, m_output_properties, n_threads));
                break;
            case SS_TEST:
                m_pipe.reset(new voc_detection_pipe(m_test_set, m_output_properties, n_threads));
                break;
            default:
                throw std::runtime_error("VOC Detection Dataset: cannot switch to supplied subset!");
        }
        m_pipe->start();
    }

    voc_detection_dataset::voc_detection_dataset(const std::string& train_filename, const std::string& test_filename, int n_threads, bool verbose)
    {
        read_meta_info(m_training_set, train_filename, verbose);
        read_meta_info(m_test_set, test_filename, verbose);
        srand(time(NULL));
        std::random_shuffle(m_training_set.begin(), m_training_set.end());
        //switch_dataset(SS_TRAIN);
    }

    void voc_detection_dataset::get_batch(std::list<pattern>& dest, unsigned int n){
        m_pipe->get_batch(dest, n);
    }
    unsigned int voc_detection_dataset::size_available()const{
        return m_pipe->size();
    }

    void voc_detection_dataset::read_meta_info(std::vector<image_meta_info>& dest, const std::string& filename, bool verbose){
        std::ifstream ifs(filename.c_str());
        unsigned int n_objs;
        unsigned int cnt_imgs=0, cnt_objs=0;
        while(ifs){
            image_meta_info imi;
            ifs >> imi.filename;
            if(imi.filename.size() == 0)
                break;
            ifs >> n_objs;
            for (unsigned int i = 0; i < n_objs; ++i)
            {
                object o;
                ifs >> o.klass;
                ifs >> o.truncated;
                ifs >> o.xmin >> o.xmax >> o.ymin >> o.ymax;
                imi.objects.push_back(o);
                cnt_objs ++;
            }
            dest.push_back(imi);
            cnt_imgs ++;
        }
        if(verbose){
            std::cout << "VOC:" << filename << " cnt_imgs:" << cnt_imgs << " cnt_objs:" << cnt_objs << std::endl;
        }
    }
    void voc_detection_dataset::save_results(std::list<pattern>& results){
        BOOST_FOREACH(pattern& pat, results){
            m_return_queue.push_back(pat);

            if(pat.meta_info.scale_id != pat.meta_info.n_scales - 1)
                continue;
            if(pat.meta_info.subimage_id != pat.meta_info.n_subimages - 1)
                continue;

            // the image has been completely processed. Collect all cropped parts.
            std::list<pattern> ready_image;

            // move patterns from m_return_queue to ready_image w/o copying (using splice)
            // http://stackoverflow.com/questions/501962/erasing-items-from-an-stl-list
            std::list<pattern>::iterator it = m_return_queue.begin();
            while (it != m_return_queue.end()) {
                bool filename_matches = it->meta_info.filename == pat.meta_info.filename;
                if (filename_matches)
                {
                    std::list<pattern>::iterator old_it = it++;
                    ready_image.splice(ready_image.end(), m_return_queue, old_it);
                }
                else
                    ++it;
            }

            // TODO: dispatch resulting image
            //std::cout << "Done with  image `" << pat.meta_info.filename << "'" << std::endl;
        }
    }
}
