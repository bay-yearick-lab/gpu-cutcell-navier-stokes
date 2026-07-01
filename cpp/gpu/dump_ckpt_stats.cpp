// Partial-stats dumper: read a running hill-DNS checkpoint and write the stats JSON
// mid-run, reusing the exact StationStats::build/load/write_json so the numbers are
// identical to what the driver would emit at the end. Host-only (no GPU), so it runs
// safely alongside a running GPU DNS. Usage: dump_ckpt_stats <ckpt.bin> <out.json>
#include <cstdio>
#include <cstdlib>
#include "hill3d.h"
#include "stats_hill.h"

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: %s <ckpt.bin> <out.json>\n", argv[0]); return 1; }
    const int ne = 32, nz = 40, p = 3;          // must match the production run
    const double Lz = 4.5, nu = 0.0001786, ub = 1.0;   // target_ub=1 (PI-controlled)
    Hill3D H; H.build(ne, p, nz, Lz, /*physical=*/true);
    StationStats stats; stats.build(H);
    const int NC = H.nr2 * nz;

    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::printf("cannot open ckpt %s\n", argv[1]); return 1; }
    int step = 0; double fx = 0, integ = 0; int nc2 = 0;
    size_t g;
    g = std::fread(&step, sizeof(int), 1, f);
    g = std::fread(&fx, sizeof(double), 1, f);
    g = std::fread(&integ, sizeof(double), 1, f);
    g = std::fread(&nc2, sizeof(int), 1, f); (void)g;
    if (nc2 != NC) std::printf("warning: NC mismatch (ckpt %d, built %d); using ckpt NC\n", nc2, NC);
    std::fseek(f, 3L * nc2 * (long)sizeof(double), SEEK_CUR);   // skip the 3 field components
    stats.load(f);
    std::fclose(f);

    const int reyn = (int)(ub / nu + 0.5);
    stats.write_json(argv[2], H, ub, reyn, step, step * 0.005, 0.0);
    std::printf("dumped step=%d nsamp=%ld -> %s\n", step, stats.n, argv[2]);
    return 0;
}
