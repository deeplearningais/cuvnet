#include <log4cxx/logger.h>
#include <boost/foreach.hpp>
#include "bounding_box_tools.hpp"
//#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#define cimg_use_jpeg
//#include <CImg.h>

namespace{
    log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("bounding_box_tools");
}

namespace cuvnet { namespace bbtools {
    void coordinate_transformer::transform(int& x, int& y)const{
    }
    void coordinate_transformer::inverse_transform(int& x, int& y)const{
    }
    rectangle 
    rectangle::scale(float factor)const{
        float cx = xmin + (xmax-xmin)/2.f;
        float cy = ymin + (ymax-ymin)/2.f;
        float dx = (cx - xmin) * factor;
        float dy = (cy - ymin) * factor;
        rectangle r;
        r.xmin = cx - dx + 0.5f;
        r.xmax = cx + dx + 0.5f;
        r.ymin = cy - dy + 0.5f;
        r.ymax = cy + dy + 0.5f;
        return r;
    }

    image::image(const std::string& fn){
        using namespace cv;
        porig.reset(new Mat(imread(fn.c_str())));
    }
    image::~image(){
    }
    image::image(const image&){/* private */}
    image& image::operator=(const image&){ /* private */ return *this; }

    void image::transpose(){
        cv::transpose(*porig, *porig);
        BOOST_FOREACH(object& o, meta.objects){
            std::swap(o.bb.xmin, o.bb.ymin);
            std::swap(o.bb.xmax, o.bb.ymax);
        }
    }

    void image::flip_lr(){
        cv::flip(*porig, *porig, 1);
        BOOST_FOREACH(object& o, meta.objects){
            o.bb.xmin = porig->cols - o.bb.xmin;
            o.bb.xmax = porig->cols - o.bb.xmax;
            o.bb.ymin = porig->cols - o.bb.ymin;
            o.bb.ymax = porig->cols - o.bb.ymax;
        }
    }

    sub_image::sub_image(const image& img, const rectangle& r)
        :original_img(img)
        ,pos(r)
        ,objfilt(NULL)
        ,scale_fact(1.f)
    {
    }

    sub_image::sub_image(const image& img)
        :original_img(img)
        ,objfilt(NULL)
        ,scale_fact(1.f)
    {
        pos.xmin = 0;
        pos.ymin = 0;
        pos.xmax = original_img.porig->cols;
        pos.ymax = original_img.porig->rows;
    }
    bool sub_image::has_objects(){
        bool found = false;
        // use the objects from the original image, no scaling required
        BOOST_FOREACH(const object& o, original_img.meta.objects){
            if(objfilt && !objfilt->filter(*this, o)) 
                continue;
            found = true;
            break;
        }
        return found;
    }
    sub_image& sub_image::crop_random_square(float frac){
        int w = pos.xmax - pos.xmin;
        int h = pos.ymax - pos.ymin;
        int sq_size = frac * std::min(w, h);
        int xoff = drand48() * (w - sq_size);
        int yoff = drand48() * (h - sq_size);
        pos.xmin += xoff;
        pos.xmax = pos.xmin + sq_size;
        pos.ymin += yoff;
        pos.ymax = pos.ymin + sq_size;
        return *this;
    }

    sub_image& sub_image::constrain_to_orig(bool clip){
        int w = pos.xmax - pos.xmin;
        int ow = original_img.porig->cols;
        if(w > ow){
            if(!clip)
                throw std::runtime_error("constrain_to_orig: subimage_width > original_width");
            w = ow;
            pos.xmin = 0;
            pos.xmax = pos.xmin + w;
        }
        int h = pos.ymax - pos.ymin;
        int oh = original_img.porig->rows;
        if(h > oh){
            if(!clip)
                throw std::runtime_error("constrain_to_orig: subimage_height > original_height");
            h = oh;
            pos.ymin = 0;
            pos.ymax = pos.ymin + h;
        }
        {   // left border
            int dleft = std::max(0,-pos.xmin);
            pos.xmin += dleft;
            pos.xmax += dleft;
        }
        {   // bottom border
            int dbot = std::max(0,-pos.ymin);
            pos.ymin += dbot;
            pos.ymax += dbot;
        }
        {   // right border
            int d = std::max(0,pos.xmax - original_img.porig->cols);
            pos.xmin -= d;
            pos.xmax -= d;
        }
        {   // top border
            int d = std::max(0,pos.ymax - original_img.porig->rows);
            pos.ymin -= d;
            pos.ymax -= d;
        }
        return *this;
    }
    sub_image& sub_image::crop(){
        if(pos.xmin < 0 || pos.ymin < 0 || pos.xmax > original_img.porig->cols || pos.ymax > original_img.porig->rows)
            throw std::runtime_error("sub_image: cannot call `crop()' on image smaller than subimage. Use square, etc.");
        pcut.reset(new cv::Mat((*original_img.porig)(cv::Range(pos.ymin, pos.ymax), cv::Range(pos.xmin, pos.xmax)).clone()));
        return *this;
    }

    sub_image& sub_image::show(const std::string name){
        if(!pcut)
            throw std::runtime_error("sub_image: cannot show, call crop() first");
        cv::namedWindow(name, CV_WINDOW_AUTOSIZE);
        cv::imshow(name, *pcut);
        while( true ){
            char c = cv::waitKey(500);
            if( (char)c == 27 ) { break; }
        }
        return *this;
    }
    sub_image& sub_image::extend_to_square(){
        int h = pos.ymax - pos.ymin;
        int w = pos.xmax - pos.xmin;
        // symmetrically extend the bounding box in both directions
        // due to rounding in the same direction always, this has
        // a bias towards moving up/left (by 1/2 pixel).
        if(h>w){
            float dy2 = (h-w)/2.f;
            pos.xmin = (int)( pos.xmin - dy2);
            pos.xmax = (int)( pos.xmax + dy2);
        }
        else if(w>h){
            float dx2 = (w-h)/2.f;
            pos.ymin = (int)( pos.ymin - dx2);
            pos.ymax = (int)( pos.ymax + dx2);
        }
        return *this;
    }

    sub_image& sub_image::remove_padding(boost::shared_ptr<cv::Mat> img, rectangle* ppos){
        if(!img) {
            if(!pcut)
                throw std::runtime_error("remove_padding: image argument NULL and no cropped subimage available!");
            img = pcut;
        }
        if(!ppos){
            ppos = &pos;
        }

        float scale_w = (ppos->xmax-ppos->xmin) / (float)(img->cols);
        float scale_h = (ppos->ymax-ppos->ymin) / (float)(img->rows);

        int dbot   = -std::min(0, ppos->ymin);
        int dleft  = -std::min(0, ppos->xmin);
        int dright = -std::min(0, -ppos->xmax + original_img.porig->cols);
        int dtop   = -std::min(0, -ppos->ymax + original_img.porig->rows);

        cv::Mat& dst = *img;

        dst = dst(
                cv::Range(dbot/scale_h+1, img->rows-dtop/scale_h),
                cv::Range(dleft/scale_w+1, img->cols-dright/scale_w)); // TODO: removed -1 for both coordinates when switching to opencv

        int new_w = (ppos->xmax-ppos->xmin) - dleft - dright;
        int new_h = (ppos->ymax-ppos->ymin) - dbot - dtop;
        ppos->xmin += dleft;
        ppos->ymin += dbot;
        ppos->xmax  = ppos->xmin + new_w;
        ppos->ymax  = ppos->ymin + new_h;
        
        return *this;
    }
    sub_image& sub_image::fill_padding(int color, cv::Mat* img, const coordinate_transformer& ct){
        if(img == NULL) {
            if(!pcut)
                throw std::runtime_error("fill_padding: image argument NULL and no cropped subimage available!");
            img = pcut.get();
        }
        cv::Mat& dst = *img;
        int dbot   = -std::min(0, pos.ymin) * scale_fact + 0.5f;
        int dleft  = -std::min(0, pos.xmin) * scale_fact + 0.5f;
        int dright = (-pos.xmin + (original_img.porig->cols - 1)) * scale_fact + 0.5f;
        int dtop   = (-pos.ymin + (original_img.porig->rows - 1)) * scale_fact + 0.5f;

        ct.transform(dleft, dbot);
        ct.transform(dright, dtop);

        if(dleft > 0)
            cv::rectangle(dst, cv::Point(0,0), cv::Point(dleft, dst.rows-1), cv::Scalar(color, color, color), CV_FILLED);
            //dst.draw_rectangle(0,0,0,0,  dleft, dst.rows-1, dst.depth()-1, dst.spectrum()-1, color);
        if(dright < dst.cols)
            cv::rectangle(dst, cv::Point(dright,0), cv::Point(dst.cols-1, dst.rows-1), cv::Scalar(color, color, color), CV_FILLED);
            //dst.draw_rectangle(dright,0,0,0,  dst.cols-1, dst.rows-1, dst.depth()-1, dst.spectrum()-1, color);
        if(dbot > 0)
            cv::rectangle(dst, cv::Point(0,0), cv::Point(dst.cols-1, dbot-1), cv::Scalar(color, color, color), CV_FILLED);
            //dst.draw_rectangle(0,0,0,0,  dst.cols-1, dbot-1, dst.depth()-1, dst.spectrum()-1, color);
        if(dtop < dst.rows)
            cv::rectangle(dst, cv::Point(0,dtop), cv::Point(dst.cols-1, dst.rows-1), cv::Scalar(color, color, color), CV_FILLED);
            //dst.draw_rectangle(0,dtop,0,0,  dst.cols-1, dst.rows-1, dst.depth()-1, dst.spectrum()-1, color);
        
        return *this;
    }
    sub_image& sub_image::crop_with_padding(int border){
        int dbot   = std::max(0, -pos.ymin);
        int dleft  = std::max(0, -pos.xmin);
        int dright = std::max(0, pos.xmax - original_img.porig->cols);
        int dtop   = std::max(0, pos.ymax - original_img.porig->rows);

        if(!dbot && !dleft && !dright && !dtop){
            crop();
            return *this;
        }
        cv::Mat dst = *original_img.porig;

        // enlarge original image
        int dh = std::max(dbot,dtop);
        int dw = std::max(dright,dleft);
        int new_h = original_img.porig->rows + 2*dh;
        int new_w = original_img.porig->cols + 2*dw;
        cv::Scalar value;
        cv::copyMakeBorder( dst, dst, dh, dh, dw, dw, cv::BORDER_REFLECT_101, value );
        cv::resize(dst, dst, cv::Size(new_w, new_h), 0.5, 0.5);

        //const unsigned char white[] = {255,255,255};
        //const unsigned char red[] = {255,0,0};
        //dst.draw_rectangle(pos.xmin, pos.ymin, pos.xmax, pos.ymax, white, 1.f, ~0U);
        //dst.draw_rectangle(pos.xmin+dw, pos.ymin+dh, pos.xmax+dw, pos.ymax+dh, red, 1.f, ~0U);

        dst = dst(cv::Range(pos.ymin+dh, pos.ymax+dh), cv::Range(pos.xmin+dw, pos.xmax+dw));

        pcut.reset(new cv::Mat(dst));

        return *this;
    }
    sub_image& sub_image::scale_larger_dim(int size){
        if(!pcut)
            throw std::runtime_error("cut first, then call scale_height!");

        scale_fact = size / (float)std::max(pcut->cols, pcut->rows);
        if(scale_fact < 1.f) {
            // blur for better subsampling quality
            cv::blur(*pcut, *pcut, cv::Size(1.f/scale_fact, 1.f/scale_fact), cv::Point(-1,-1), cv::BORDER_REFLECT_101);
        }
        cv::resize(*pcut, *pcut, cv::Size(size, size), 0.5f, 0.5f);
        //pcut->resize(size, size, -100, -100, 3, 2, 0.5f, 0.5f);
        return *this;
    }

    std::vector<std::vector<rectangle> > 
    sub_image::get_objects(int n_classes, float scale)const{
        std::vector<std::vector<rectangle> > bboxes;
        const std::vector<object>& objs = original_img.meta.objects;
        bboxes.resize(n_classes);
        BOOST_FOREACH(const object& orig_o, objs){
            if(objfilt && !objfilt->filter(*this, orig_o)) 
                continue;
            object o = object_relative_to_subimg(orig_o);
            o.bb = o.bb.scale(scale);
            assert(n_classes > (int)o.klass);
            bboxes[o.klass].push_back(o.bb);
            //bboxes[0].push_back(o.bb); // only one class for now....
        }
        return bboxes;
    }
    
    sub_image& 
    sub_image::mark_objects(int type, unsigned char color, float scale, boost::shared_ptr<cv::Mat> img, std::vector<std::vector<bbtools::rectangle> >* bboxes){
        if(!img){
            if(!pcut)
                throw std::runtime_error("mark_objects: supply image or crop first!");
            img = pcut;
        }
        const std::vector<object>& objs = original_img.meta.objects;
        if(type==2)
            img->setTo(0);
        if(bboxes)
            bboxes->resize(img->depth());
        BOOST_FOREACH(const object& orig_o, objs){
            if(objfilt && !objfilt->filter(*this, orig_o)) 
                continue;
            object o = object_relative_to_subimg(orig_o);
            if(bboxes){
                //(*bboxes)[o.klass].push_back(o.bb);
                (*bboxes)[0].push_back(o.bb); // only one class for now....
            }
            //if(type==0){
                cv::rectangle(*img, cv::Point(o.bb.xmin, o.bb.ymin), cv::Point(o.bb.xmax, o.bb.ymax), cv::Scalar(color, color, color), CV_FILLED); 
                //img->draw_rectangle(o.bb.xmin, o.bb.ymin, o.bb.xmax, o.bb.ymax, clr, 1.f, ~0U);
            //}else{
            //    cuvAssert(false);
            //}
            /*
             *else if(type == 1){
             *    img->draw_rectangle(o.bb.xmin, o.bb.ymin,0,0, o.bb.xmax, o.bb.ymax, img->depth()-1, img->spectrum()-1, color);
             *}else if(type == 2){
             *    const float cx = (o.bb.xmax+o.bb.xmin)/2.f;
             *    const float cy = (o.bb.ymax+o.bb.ymin)/2.f;
             *    float w  = o.bb.xmax - o.bb.xmin + 1;
             *    float h  = o.bb.ymax - o.bb.ymin + 1;
             *    w *= scale;
             *    h *= scale;
             *    w *= w; // square std-dev
             *    h *= h; // square std-dev
             *    cimg_forXY(*img,dx,dy) {
             *        float mahalanobis_dist = sqrt(
             *                (dx-cx)*(dx-cx)/w +
             *                (dy-cy)*(dy-cy)/h);
             *        (*img)(dx,dy) = std::max((*img)(dx,dy), (unsigned char) (255*exp(-0.5*mahalanobis_dist)));
             *    }
             *}
             */
        }
        return *this;
    }

    object sub_image::object_relative_to_subimg(const object& o)const{
        object r = o;
        r.bb.xmin = (o.bb.xmin-pos.xmin) * scale_fact;
        r.bb.xmax = (o.bb.xmax-pos.xmin) * scale_fact;
        r.bb.ymin = (o.bb.ymin-pos.ymin) * scale_fact;
        r.bb.ymax = (o.bb.ymax-pos.ymin) * scale_fact;
        return r;
    }

    sub_image::~sub_image(){
    }

    bool
    object_filter::filter(const sub_image& si, const object& o){
        if(o.bb.xmax < si.pos.xmin) return false;
        if(o.bb.ymax < si.pos.ymin) return false;

        if(o.bb.xmin > si.pos.xmax) return false;
        if(o.bb.ymin > si.pos.ymax) return false;
        return true;
    }
    bool
    single_class_object_filter::filter(const sub_image& si, const object& o){
        if(o.klass != klass) // 14: person
            return false;;
        return object_filter::filter(si,o);
    }
    bool similar_scale_object_filter::filter(const sub_image& si, const object& o){
        int w = si.pos.xmax - si.pos.xmin + 1;
        int ow = o.bb.xmax - o.bb.xmin + 1;
        if(ow < min_frac*w) return false;
        if(ow > max_frac*w) return false;

        int h = si.pos.ymax - si.pos.ymin + 1;
        int oh = o.bb.ymax - o.bb.ymin + 1;
        if(oh < min_frac*h) return false;
        if(oh > max_frac*h) return false;
        return object_filter::filter(si,o);
    }
} }
