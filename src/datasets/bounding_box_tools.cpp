#include <log4cxx/logger.h>
#include <boost/foreach.hpp>
#include "bounding_box_tools.hpp"
#include <cuv/libs/cimg/cuv_cimg.hpp>
#define cimg_use_jpeg
#include <CImg.h>

namespace{
    log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("bounding_box_tools");
}

namespace cuvnet { namespace bbtools {
    image::image(const std::string& fn){
        using namespace cimg_library;
        porig = new CImg<unsigned char>(fn.c_str());
    }
    image::~image(){
        delete porig;
    }
    image::image(const image&){/* private */}
    image& image::operator=(const image&){ /* private */ return *this; }

    void image::transpose(){
        porig->transpose();
        BOOST_FOREACH(object& o, meta.objects){
            std::swap(o.bb.xmin, o.bb.ymin);
            std::swap(o.bb.xmax, o.bb.ymax);
        }
    }

    sub_image::sub_image(const image& img, const rectangle& r)
        :original_img(img)
        ,pcut(NULL)
        ,pos(r)
    {
    }

    sub_image::sub_image(const image& img)
        :original_img(img)
        ,pcut(NULL)
    {
        pos.xmin = 0;
        pos.ymin = 0;
        pos.xmax = original_img.porig->width();
        pos.ymax = original_img.porig->height();
    }
    bool sub_image::has_objects(){
        std::vector<object> objs = objects();
        bool found = false;
        BOOST_FOREACH(const object& o, objs){
            if(o.bb.xmin > pos.xmax)
                continue;
            if(o.bb.ymin > pos.ymax)
                continue;
            if(o.bb.xmax < 0)
                continue;
            if(o.bb.ymax < 0)
                continue;
            found = true;
            break;
        }
        return found;
    }

    sub_image& sub_image::constrain_to_orig(bool clip){
        int w = pos.xmax - pos.xmin;
        int ow = original_img.porig->width();
        if(w > ow){
            if(!clip)
                throw std::runtime_error("constrain_to_orig: subimage_width > original_width");
            w = ow;
            pos.xmax = pos.xmin + w -1;
        }
        int h = pos.ymax - pos.ymin;
        int oh = original_img.porig->height();
        if(h > oh){
            if(!clip)
                throw std::runtime_error("constrain_to_orig: subimage_height > original_height");
            h = oh;
            pos.ymax = pos.ymin + h -1;
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
            int d = std::max(0,pos.xmax - original_img.porig->width() + 1);
            pos.xmin -= d;
            pos.xmax -= d;
        }
        {   // top border
            int d = std::max(0,pos.ymax - original_img.porig->height() + 1);
            pos.ymin -= d;
            pos.ymax -= d;
        }
        return *this;
    }
    sub_image& sub_image::crop(){
        if(pcut)
            delete pcut;
        if(pos.xmin < 0 || pos.ymin < 0 || pos.xmax >= original_img.porig->width() || pos.ymax >= original_img.porig->height())
            throw std::runtime_error("sub_image: cannot call `crop()' on image smaller than subimage. Use square, etc.");
        pcut = new cimg_library::CImg<unsigned char>(original_img.porig->get_crop(pos.xmin, pos.ymin, pos.xmax, pos.ymax, false));
        return *this;
    }

    sub_image& sub_image::show(const std::string name){
        if(!pcut)
            throw std::runtime_error("sub_image: cannot show, call crop() first");
        cimg_library::CImgDisplay main_disp(*pcut, name.c_str());
        main_disp.wait();
        return *this;
    }
    sub_image& sub_image::extend_to_square(){
        int h = pos.ymax - pos.ymin;
        int w = pos.xmax - pos.xmin;
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

    sub_image& sub_image::remove_padding(cimg_library::CImg<unsigned char>* img){
        if(img == NULL) {
            if(!pcut)
                throw std::runtime_error("remove_padding: image argument NULL and no cropped subimage available!");
            img = pcut;
        }
        float scale_w = (pos.xmax-pos.xmin+1) / (float)(img->width());
        float scale_h = (pos.ymax-pos.ymin+1) / (float)(img->height());

        int dbot   = -std::min(0, pos.ymin);
        int dleft  = -std::min(0, pos.xmin);
        int dright = -std::min(0, -pos.xmax + original_img.porig->width() - 1);
        int dtop   = -std::min(0, -pos.ymax + original_img.porig->height() - 1);

        cimg_library::CImg<unsigned char>& dst = *img;

        dst.crop(dleft/scale_w+1, dbot/scale_h+1,
                img->width()-dright/scale_w-1, img->height()-dtop/scale_h-1, false);

        int new_w = (pos.xmax-pos.xmin) - dleft - dright;
        int new_h = (pos.ymax-pos.ymin) - dbot - dtop;
        pos.xmin += dleft;
        pos.xmax += dleft;
        pos.ymin += dbot;
        pos.ymax += dbot;
        pos.xmax  = pos.xmin + new_w;
        pos.ymax  = pos.ymin + new_h;
        
        return *this;
    }
    sub_image& sub_image::fill_padding(int color, cimg_library::CImg<unsigned char>* img){
        if(img == NULL) {
            if(!pcut)
                throw std::runtime_error("fill_padding: image argument NULL and no cropped subimage available!");
            img = pcut;
        }
        float scale_w = (pos.xmax-pos.xmin+1) / (float)(img->width());
        float scale_h = (pos.ymax-pos.ymin+1) / (float)(img->height());

        int dbot   = -std::min(0, pos.ymin) / scale_h;
        int dleft  = -std::min(0, pos.xmin) / scale_w;
        int dright = -std::min(0, -pos.xmax + original_img.porig->width() - 1) / scale_w;
        int dtop   = -std::min(0, -pos.ymax + original_img.porig->height() - 1) / scale_h;

        cimg_library::CImg<unsigned char>& dst = *img;

        if(dleft > 0)
            dst.draw_rectangle(0,0,0,0,  dleft, dst.height()-1, dst.depth()-1, dst.spectrum()-1, color);
        if(dright > 0)
            dst.draw_rectangle(dst.width()-dright,0,0,0,  dst.width()-1, dst.height()-1, dst.depth()-1, dst.spectrum()-1, color);
        if(dbot > 0)
            dst.draw_rectangle(0,0,0,0,  dst.width()-1, dbot-1, dst.depth()-1, dst.spectrum()-1, color);
        if(dtop > 0)
            dst.draw_rectangle(0,dst.height()-dtop,0,0,  dst.width()-1, dst.height()-1, dst.depth()-1, dst.spectrum()-1, color);
        
        return *this;
    }
    sub_image& sub_image::crop_with_padding(int border){
        int dbot   = std::max(0, -pos.ymin);
        int dleft  = std::max(0, -pos.xmin);
        int dright = std::max(0, pos.xmax - original_img.porig->width() + 1);
        int dtop   = std::max(0, pos.ymax - original_img.porig->height() + 1);

        if(!dbot && !dleft && !dright && !dtop){
            crop();
            return *this;
        }
        cimg_library::CImg<unsigned char> dst = *original_img.porig;

        // enlarge original image
        int dh = std::max(dbot,dtop);
        int dw = std::max(dright,dleft);
        int new_h = original_img.porig->height() + 2*dh;
        int new_w = original_img.porig->width() + 2*dw;
        dst.resize(new_w, new_h, -100, -100, 0, border, 0.5f, 0.5f);

        //const unsigned char white[] = {255,255,255};
        //const unsigned char red[] = {255,0,0};
        //dst.draw_rectangle(pos.xmin, pos.ymin, pos.xmax, pos.ymax, white, 1.f, ~0U);
        //dst.draw_rectangle(pos.xmin+dw, pos.ymin+dh, pos.xmax+dw, pos.ymax+dh, red, 1.f, ~0U);

        dst.crop(pos.xmin+dw, pos.ymin+dh, pos.xmax+dw, pos.ymax+dh, false);

        if(pcut)
            delete pcut;
        pcut = new cimg_library::CImg<unsigned char>(dst);

        return *this;
    }
    sub_image& sub_image::scale_larger_dim(int size){
        if(!pcut)
            throw std::runtime_error("cut first, then call scale_height!");

        //float fact = size / (float)std::max(pcut->width(), pcut->height());
        //pcut->resize(fact * pcut->width(), fact * pcut->height(), -100, -100, 3, 2, 0.5f, 0.5f);
        pcut->resize(size, size, -100, -100, 3, 2, 0.5f, 0.5f);
        return *this;
    }

    
    sub_image& 
    sub_image::mark_objects(int type, unsigned char color, float scale, cimg_library::CImg<unsigned char>* img1){
        if(!img1){
            if(!pcut)
                throw std::runtime_error("mark_objects: supply image or crop first!");
            img1 = pcut;
        }
        std::vector<object> objs = objects();
        const unsigned char clr[] = {color,color,color};
        if(type==2)
            img1->fill(0);
        BOOST_FOREACH(const object& o, objs){
            if(type==0){
                img1->draw_rectangle(o.bb.xmin, o.bb.ymin, o.bb.xmax, o.bb.ymax, clr, 1.f, ~0U);
            }else if(type == 1){
                img1->draw_rectangle(o.bb.xmin, o.bb.ymin,0,0, o.bb.xmax, o.bb.ymax, img1->depth()-1, img1->spectrum()-1, color);
            }else if(type == 2){
                const float cx = (o.bb.xmax+o.bb.xmin)/2.f;
                const float cy = (o.bb.ymax+o.bb.ymin)/2.f;
                float w  = o.bb.xmax - o.bb.xmin + 1;
                float h  = o.bb.ymax - o.bb.ymin + 1;
                float w1 = w * scale;
                float h1 = h * scale;
                cimg_forXY(*img1,dx,dy) {
                    float mahalanobis_dist = sqrt(
                            (dx-cx)*(dx-cx)/w1 +
                            (dy-cy)*(dy-cy)/h1);
                    (*img1)(dx,dy) = std::max((*img1)(dx,dy), (unsigned char) (255*exp(-0.5*mahalanobis_dist)));
                }
            }
        }
        return *this;
    }

    std::vector<object> 
    sub_image::objects(int size){
        std::vector<object> res;
        int orig_w = original_img.porig->width();
        int orig_h = original_img.porig->height();
        float scale_w = (pos.xmax-pos.xmin);
        float scale_h = (pos.ymax-pos.ymin);
        if(size < 0){
            if(pcut){
                scale_w /= pcut->width();
                scale_h /= pcut->height();
            }
        }else if(size>0){
                scale_w /= size;
                scale_h /= size;
        }
        BOOST_FOREACH(object o, original_img.meta.objects){
            o.bb.xmin = (o.bb.xmin-pos.xmin) / scale_w;
            o.bb.xmax = (o.bb.xmax-pos.xmin) / scale_w;
            o.bb.ymin = (o.bb.ymin-pos.ymin) / scale_h;
            o.bb.ymax = (o.bb.ymax-pos.ymin) / scale_h;
            res.push_back(o);
        }
        return res;
    }

    sub_image::~sub_image(){
        delete pcut;
    }
} }
