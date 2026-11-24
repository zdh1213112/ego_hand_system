// Rasterize the front surface of the fitted hand in crop pixels.
#include <algorithm>
#include <cmath>
extern "C" void render(int width, int faces, const int* index, const float* xyz, float* depth) {
    for(int i=0;i<faces;++i) {
        const float *a=xyz+3*index[3*i],*b=xyz+3*index[3*i+1],*c=xyz+3*index[3*i+2];
        const float den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
        if(!std::isfinite(den)||std::abs(den)<1.e-6f||std::min({a[2],b[2],c[2]})<=0) continue;
        int x0=std::max(0,int(std::floor(std::min({a[0],b[0],c[0]}))));
        int x1=std::min(width-1,int(std::ceil(std::max({a[0],b[0],c[0]}))));
        int y0=std::max(0,int(std::floor(std::min({a[1],b[1],c[1]}))));
        int y1=std::min(width-1,int(std::ceil(std::max({a[1],b[1],c[1]}))));
        for(int y=y0;y<=y1;y++) for(int x=x0;x<=x1;x++) {
            float u=((b[1]-c[1])*(x+.5f-c[0])+(c[0]-b[0])*(y+.5f-c[1]))/den;
            float v=((c[1]-a[1])*(x+.5f-c[0])+(a[0]-c[0])*(y+.5f-c[1]))/den;
            if(u<0||v<0||u+v>1) continue;
            float z=1.f/(u/a[2]+v/b[2]+(1-u-v)/c[2]);
            depth[y*width+x]=std::min(depth[y*width+x],z);
        }
    }
}
