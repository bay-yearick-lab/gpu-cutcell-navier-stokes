// vtk.h
//
// Dependency-free VTK XML output for ParaView. Writes structured-grid
// ImageData (.vti) with raw appended Float32 binary, plus a .pvd time-series
// collection. ParaView reads these natively (volume render, Q/omega
// isosurfaces, slices). No VTK library, no exceptions -- just fwrite.
//
// Point ordering in VTK ImageData is x-fastest, which is EXACTLY the Field3D
// layout (idx = (k*ny + j)*nx + i), so fields stream out with no transpose.
// We emit Float32 (visualization precision) to halve file size.

#ifndef VTK_IO_H
#define VTK_IO_H

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/stat.h>   // mkdir

namespace vtk {

// One named scalar field over the grid (length nx*ny*nz, x-fastest).
struct NamedField {
    std::string name;
    const std::vector<double>* data;
};

inline void make_dir(const std::string& d) { ::mkdir(d.c_str(), 0755); }

// Write a single .vti frame.
inline void write_vti(const std::string& path,
                      int nx, int ny, int nz,
                      double dx, double dy, double dz,
                      const std::vector<NamedField>& fields,
                      double ox = 0.0, double oy = 0.0, double oz = 0.0) {
    const uint64_t npts = (uint64_t)nx * ny * nz;
    std::string h;
    char buf[640];
    h += "<?xml version=\"1.0\"?>\n";
    h += "<VTKFile type=\"ImageData\" version=\"1.0\" "
         "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    std::snprintf(buf, sizeof buf,
        "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"%g %g %g\" "
        "Spacing=\"%g %g %g\">\n",
        nx - 1, ny - 1, nz - 1, ox, oy, oz, dx, dy, dz);
    h += buf;
    std::snprintf(buf, sizeof buf, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n",
                  nx - 1, ny - 1, nz - 1);
    h += buf;
    h += "      <PointData Scalars=\"" +
         (fields.empty() ? std::string() : fields[0].name) + "\">\n";
    uint64_t off = 0;
    for (const auto& f : fields) {
        std::snprintf(buf, sizeof buf,
            "        <DataArray type=\"Float32\" Name=\"%s\" "
            "NumberOfComponents=\"1\" format=\"appended\" offset=\"%llu\"/>\n",
            f.name.c_str(), (unsigned long long)off);
        h += buf;
        off += sizeof(uint64_t) + npts * sizeof(float);
    }
    h += "      </PointData>\n";
    h += "    </Piece>\n";
    h += "  </ImageData>\n";
    h += "  <AppendedData encoding=\"raw\">\n_";

    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;
    std::fwrite(h.data(), 1, h.size(), fp);
    std::vector<float> tmp(npts);
    for (const auto& f : fields) {
        const uint64_t nbytes = npts * sizeof(float);
        std::fwrite(&nbytes, sizeof(uint64_t), 1, fp);
        const std::vector<double>& d = *f.data;
        for (uint64_t i = 0; i < npts; ++i) tmp[i] = (float)d[i];
        std::fwrite(tmp.data(), sizeof(float), npts, fp);
    }
    const std::string tail = "\n  </AppendedData>\n</VTKFile>\n";
    std::fwrite(tail.data(), 1, tail.size(), fp);
    std::fclose(fp);
}

// A .pvd time series: writes one .vti per frame into `dir`, then a collection
// file `dir/base.pvd` that ParaView opens as an animation.
struct PvdSeries {
    std::string dir;
    std::string base;
    std::vector<std::pair<double, std::string>> frames;   // (time, filename)

    PvdSeries(std::string dir_, std::string base_)
        : dir(std::move(dir_)), base(std::move(base_)) { make_dir(dir); }

    void add_frame(double time, int nx, int ny, int nz,
                   double dx, double dy, double dz,
                   const std::vector<NamedField>& fields) {
        char name[256];
        std::snprintf(name, sizeof name, "%s_%04d.vti",
                      base.c_str(), (int)frames.size());
        write_vti(dir + "/" + name, nx, ny, nz, dx, dy, dz, fields);
        frames.emplace_back(time, std::string(name));
    }

    void write_pvd() const {
        std::string s =
            "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"Collection\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n  <Collection>\n";
        char buf[512];
        for (const auto& fr : frames) {
            std::snprintf(buf, sizeof buf,
                "    <DataSet timestep=\"%g\" group=\"\" part=\"0\" file=\"%s\"/>\n",
                fr.first, fr.second.c_str());
            s += buf;
        }
        s += "  </Collection>\n</VTKFile>\n";
        FILE* fp = std::fopen((dir + "/" + base + ".pvd").c_str(), "wb");
        if (!fp) return;
        std::fwrite(s.data(), 1, s.size(), fp);
        std::fclose(fp);
    }
};

}  // namespace vtk

#endif  // VTK_IO_H
