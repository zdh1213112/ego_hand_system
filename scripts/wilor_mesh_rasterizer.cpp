// Small CPU depth-buffer renderer. No dependency on Python or OpenGL.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" void render_wilor_mesh(
    uint8_t* image, int width, int height, const float* vertices,
    const float* pixels, int vertex_count, const int32_t* faces,
    int face_count, const float* color, float alpha, int wire) {
  float min_x = float(width), min_y = float(height), max_x = -1, max_y = -1;
  for (int i = 0; i < vertex_count; ++i) {
    min_x = std::min(min_x, pixels[2*i]);
    max_x = std::max(max_x, pixels[2*i]);
    min_y = std::min(min_y, pixels[2*i+1]);
    max_y = std::max(max_y, pixels[2*i+1]);
  }
  int x0 = std::max(0, int(std::floor(std::max(-100000.f, min_x))));
  int y0 = std::max(0, int(std::floor(std::max(-100000.f, min_y))));
  int x1 = std::min(width-1, int(std::ceil(std::min(100000.f, max_x))));
  int y1 = std::min(height-1, int(std::ceil(std::min(100000.f, max_y))));
  if (x0 > x1 || y0 > y1) return;
  const int rw = x1-x0+1, rh = y1-y0+1;
  std::vector<float> depth(rw*rh, std::numeric_limits<float>::infinity());
  std::vector<uint8_t> surface(rw*rh*3);
  const float light_norm = std::sqrt(.3f*.3f+.5f*.5f+.8f*.8f);
  for (int face = 0; face < face_count; ++face) {
    const int a=faces[3*face], b=faces[3*face+1], c=faces[3*face+2];
    const float* va=vertices+3*a; const float* vb=vertices+3*b;
    const float* vc=vertices+3*c;
    if (std::min({va[2],vb[2],vc[2]}) <= .01f) continue;
    const float ax=pixels[2*a], ay=pixels[2*a+1];
    const float bx=pixels[2*b], by=pixels[2*b+1];
    const float cx=pixels[2*c], cy=pixels[2*c+1];
    const float area=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy);
    if (std::abs(area)<1e-9f) continue;
    const float ux=vb[0]-va[0], uy=vb[1]-va[1], uz=vb[2]-va[2];
    const float vx=vc[0]-va[0], vy=vc[1]-va[1], vz=vc[2]-va[2];
    const float nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
    const float norm=std::sqrt(nx*nx+ny*ny+nz*nz);
    if (norm<1e-10f) continue;
    const float light=std::abs(-.3f*nx-.5f*ny-.8f*nz)/(norm*light_norm);
    uint8_t shade[3];
    for (int k=0;k<3;++k)
      shade[k]=uint8_t(std::min(255.f,color[k]*(.40f+.60f*light)+10.f));
    const int left=std::max(x0, int(std::floor(std::max(-100000.f,std::min({ax,bx,cx})))));
    const int right=std::min(x1, int(std::ceil(std::min(100000.f,std::max({ax,bx,cx})))));
    const int top=std::max(y0, int(std::floor(std::max(-100000.f,std::min({ay,by,cy})))));
    const int bottom=std::min(y1, int(std::ceil(std::min(100000.f,std::max({ay,by,cy})))));
    for (int y=top;y<=bottom;++y) for (int x=left;x<=right;++x) {
      const float px=x+.5f, py=y+.5f;
      const float wa=((by-cy)*(px-cx)+(cx-bx)*(py-cy))/area;
      const float wb=((cy-ay)*(px-cx)+(ax-cx)*(py-cy))/area;
      const float wc=1.f-wa-wb;
      if (std::min({wa,wb,wc}) < -1e-6f) continue;
      const float z=1.f/(wa/va[2]+wb/vb[2]+wc/vc[2]);
      const int offset=(y-y0)*rw+x-x0;
      if (z >= depth[offset]) continue;
      depth[offset]=z;
      const bool edge=wire && face%9==0 && std::min({wa,wb,wc})<.04f;
      for (int k=0;k<3;++k)
        surface[3*offset+k]=edge ? uint8_t(color[k]*.55f) : shade[k];
    }
  }
  for (int y=y0;y<=y1;++y) for (int x=x0;x<=x1;++x) {
    const int offset=(y-y0)*rw+x-x0;
    if (!std::isfinite(depth[offset])) continue;
    for (int k=0;k<3;++k) {
      const int target=3*(y*width+x)+k;
      image[target]=uint8_t(std::lround((1.f-alpha)*image[target]+alpha*surface[3*offset+k]));
    }
  }
}
