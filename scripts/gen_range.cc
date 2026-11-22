#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct RangeI32 {
	int32_t l;
	int32_t u;
};

static constexpr int32_t kSift1MAttrMin = 0;
static constexpr int32_t kSift1MAttrMax = 999999;
static constexpr int kMixedFractionMinPow = 1;
static constexpr int kMixedFractionMaxPow = 10;

static auto ReadRangesIbin(const std::string& path) -> std::vector<RangeI32> {
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs.is_open()) {
		throw std::runtime_error("failed to open file: " + path);
	}

	ifs.seekg(0, std::ios::end);
	const std::streamoff file_size = ifs.tellg();
	ifs.seekg(0, std::ios::beg);

	if (file_size < 0 || (file_size % static_cast<std::streamoff>(sizeof(RangeI32)) != 0)) {
		throw std::runtime_error("invalid .ibin size, not multiple of 8 bytes");
	}

	const size_t n = static_cast<size_t>(file_size / static_cast<std::streamoff>(sizeof(RangeI32)));
	std::vector<RangeI32> ranges(n);
	ifs.read(reinterpret_cast<char*>(ranges.data()), static_cast<std::streamsize>(n * sizeof(RangeI32)));
	if (!ifs) {
		throw std::runtime_error("failed to read complete .ibin content");
	}
	return ranges;
}

static void WriteRangesIbin(const std::string& path, const std::vector<RangeI32>& ranges) {
	std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
	if (!ofs.is_open()) {
		throw std::runtime_error("failed to open output file: " + path);
	}
	ofs.write(reinterpret_cast<const char*>(ranges.data()),
						static_cast<std::streamsize>(ranges.size() * sizeof(RangeI32)));
	if (!ofs) {
		throw std::runtime_error("failed to write output .ibin");
	}
}

static auto Percentile(std::vector<int64_t> vals, double p) -> int64_t {
	if (vals.empty()) {
		return 0;
	}
	if (p <= 0.0) {
		return *std::min_element(vals.begin(), vals.end());
	}
	if (p >= 1.0) {
		return *std::max_element(vals.begin(), vals.end());
	}
	std::sort(vals.begin(), vals.end());
	const size_t idx = static_cast<size_t>(p * static_cast<double>(vals.size() - 1));
	return vals[idx];
}

static auto CeilDiv(int64_t x, int64_t y) -> int64_t {
	if (y <= 0) {
		throw std::runtime_error("divisor must be positive");
	}
	return (x + y - 1) / y;
}

static auto GenerateMixedFractionRanges(size_t count,
		int32_t attr_min,
		int32_t attr_max,
		uint64_t seed) -> std::vector<RangeI32> {
	if (attr_min > attr_max) {
		throw std::runtime_error("attr_min must be <= attr_max");
	}

	const int64_t attr_count = static_cast<int64_t>(attr_max) - static_cast<int64_t>(attr_min) + 1;
	if (attr_count <= 0) {
		throw std::runtime_error("attribute domain must be positive");
	}

	const size_t fraction_bucket_num =
			static_cast<size_t>(kMixedFractionMaxPow - kMixedFractionMinPow + 1);
	std::vector<size_t> bucket_counts(fraction_bucket_num, count / fraction_bucket_num);
	for (size_t i = 0; i < count % fraction_bucket_num; ++i) {
		++bucket_counts[i];
	}

	std::mt19937_64 rng(seed);
	std::vector<RangeI32> out;
	out.reserve(count);

	for (size_t bucket = 0; bucket < fraction_bucket_num; ++bucket) {
		const int exponent = kMixedFractionMinPow + static_cast<int>(bucket);
		const int64_t denominator = static_cast<int64_t>(1) << exponent;
		const int64_t width = std::max<int64_t>(1, CeilDiv(attr_count, denominator));
		const int64_t max_start = static_cast<int64_t>(attr_max) - width + 1;
		std::uniform_int_distribution<int64_t> start_dist(attr_min, std::max<int64_t>(attr_min, max_start));

		for (size_t i = 0; i < bucket_counts[bucket]; ++i) {
			const int64_t l = start_dist(rng);
			const int64_t u = std::min<int64_t>(attr_max, l + width - 1);
			out.push_back(RangeI32{static_cast<int32_t>(l), static_cast<int32_t>(u)});
		}
	}

	std::shuffle(out.begin(), out.end(), rng);
	return out;
}

// Single-selectivity ranges: every query gets the SAME 2^-exponent width.
// Uses the identical width/start formula as GenerateMixedFractionRanges' bucket
// for that exponent, so a `gen_sel <p>` file is directly comparable with the 2^-p
// bucket inside a mixed file -- only the query count per width differs.
static auto GenerateFixedFractionRanges(size_t count,
		int exponent,
		int32_t attr_min,
		int32_t attr_max,
		uint64_t seed) -> std::vector<RangeI32> {
	if (attr_min > attr_max) {
		throw std::runtime_error("attr_min must be <= attr_max");
	}
	if (exponent < 0 || exponent > 30) {
		throw std::runtime_error("exponent must be in [0, 30]");
	}

	const int64_t attr_count = static_cast<int64_t>(attr_max) - static_cast<int64_t>(attr_min) + 1;
	if (attr_count <= 0) {
		throw std::runtime_error("attribute domain must be positive");
	}

	const int64_t denominator = static_cast<int64_t>(1) << exponent;
	const int64_t width = std::max<int64_t>(1, CeilDiv(attr_count, denominator));
	const int64_t max_start = static_cast<int64_t>(attr_max) - width + 1;

	std::mt19937_64 rng(seed);
	std::uniform_int_distribution<int64_t> start_dist(attr_min, std::max<int64_t>(attr_min, max_start));

	std::vector<RangeI32> out;
	out.reserve(count);
	for (size_t i = 0; i < count; ++i) {
		const int64_t l = start_dist(rng);
		const int64_t u = std::min<int64_t>(attr_max, l + width - 1);
		out.push_back(RangeI32{static_cast<int32_t>(l), static_cast<int32_t>(u)});
	}
	return out;
}

int main(int argc, char** argv) {
	try {
		// Single-selectivity mode:
		//   gen_range gen_sel <output_path> <exponent> [count] [attr_min] [attr_max] [seed]
		// Emits `count` ranges all of width ceil(attr_count / 2^exponent), i.e. a
		// pure 2^-exponent selectivity file (vs `gen`, which mixes 2^-1..2^-10).
		if (argc >= 2 && std::string(argv[1]) == "gen_sel") {
			if (argc < 4) {
				std::cerr << "Usage: gen_range gen_sel <output_path> <exponent>"
						  << " [count] [attr_min] [attr_max] [seed]\n"
						  << "  exponent: selectivity is 2^-exponent (e.g. 1, 4, 7, 10)\n";
				return 1;
			}
			const std::string out_path = argv[2];
			const int exponent = std::stoi(argv[3]);
			const size_t count = (argc >= 5) ? static_cast<size_t>(std::stoull(argv[4])) : 10000;
			const int32_t global_min = (argc >= 6) ? static_cast<int32_t>(std::stoll(argv[5])) : kSift1MAttrMin;
			const int32_t global_max = (argc >= 7) ? static_cast<int32_t>(std::stoll(argv[6])) : kSift1MAttrMax;
			const uint64_t seed = (argc >= 8) ? static_cast<uint64_t>(std::stoull(argv[7])) : std::random_device{}();

			if (global_min > global_max) {
				throw std::runtime_error("global_min must be <= global_max");
			}

			const auto out = GenerateFixedFractionRanges(count, exponent, global_min, global_max, seed);
			WriteRangesIbin(out_path, out);

			const int64_t attr_count = static_cast<int64_t>(global_max) - static_cast<int64_t>(global_min) + 1;
			const int64_t width = std::max<int64_t>(1, CeilDiv(attr_count, static_cast<int64_t>(1) << exponent));
			std::cout << "Generated .ibin: " << out_path << "\n";
			std::cout << "selectivity=2^-" << exponent << " width=" << width
					  << " count=" << count
					  << " attr_range=[" << global_min << ", " << global_max << "]"
					  << " seed=" << seed << "\n";
			return 0;
		}

		if (argc == 1 || (argc >= 2 && std::string(argv[1]) == "gen")) {
			// Usage:
			// gen_range
			// gen_range gen <output_path> [count] [attr_min] [attr_max] [seed]
			// Mixed range fraction: 2^-1 to 2^-10, query count evenly distributed across 10 buckets.
			// Args are at fixed positions: the only two accepted forms are
			// `gen_range` (argc==1, all defaults) and `gen_range gen [path] [count]
			// [min] [max] [seed]`, so the optional args always start at argv[2].
			// (A previous arg_offset shifted every read by one when "gen" was
			// present, making `gen_range gen <path> <count> ...` read the path from
			// argv[3] -- explicit args silently landed in the wrong parameters.)
			const std::string out_path = (argc >= 3) ? argv[2] : "data/Sift1m/sift_query_range.ibin";
			const size_t count = (argc >= 4) ? static_cast<size_t>(std::stoull(argv[3])) : 10000;
			const int32_t global_min = (argc >= 5) ? static_cast<int32_t>(std::stoll(argv[4])) : kSift1MAttrMin;
			const int32_t global_max = (argc >= 6) ? static_cast<int32_t>(std::stoll(argv[5])) : kSift1MAttrMax;
			const uint64_t seed = (argc >= 7) ? static_cast<uint64_t>(std::stoull(argv[6])) : std::random_device{}();

			if (global_min > global_max) {
				throw std::runtime_error("global_min must be <= global_max");
			}

			const auto out = GenerateMixedFractionRanges(count, global_min, global_max, seed);
			WriteRangesIbin(out_path, out);

			std::cout << "Generated .ibin: " << out_path << "\n";
			std::cout << "count=" << count << " attr_range=[" << global_min << ", " << global_max << "] seed=" << seed << "\n";
			std::cout << "Mixed range fraction buckets:\n";
			const int64_t attr_count = static_cast<int64_t>(global_max) - static_cast<int64_t>(global_min) + 1;
			const size_t fraction_bucket_num =
					static_cast<size_t>(kMixedFractionMaxPow - kMixedFractionMinPow + 1);
			std::vector<size_t> bucket_counts(fraction_bucket_num, count / fraction_bucket_num);
			for (size_t i = 0; i < count % fraction_bucket_num; ++i) {
				++bucket_counts[i];
			}
			for (size_t bucket = 0; bucket < fraction_bucket_num; ++bucket) {
				const int exponent = kMixedFractionMinPow + static_cast<int>(bucket);
				const int64_t denominator = static_cast<int64_t>(1) << exponent;
				const int64_t width = std::max<int64_t>(1, CeilDiv(attr_count, denominator));
				std::cout << "  2^-" << exponent << ": " << bucket_counts[bucket]
						  << " queries, width=" << width << "\n";
			}
			return 0;
		}

		const std::string path = argv[1];
		const auto ranges = ReadRangesIbin(path);

		if (ranges.empty()) {
			std::cout << "Empty range file: " << path << "\n";
			return 0;
		}

		int32_t min_l = std::numeric_limits<int32_t>::max();
		int32_t max_l = std::numeric_limits<int32_t>::min();
		int32_t min_u = std::numeric_limits<int32_t>::max();
		int32_t max_u = std::numeric_limits<int32_t>::min();
		int64_t min_w = std::numeric_limits<int64_t>::max();
		int64_t max_w = std::numeric_limits<int64_t>::min();

		size_t invalid = 0;
		size_t point_ranges = 0;
		std::vector<int64_t> widths;
		widths.reserve(ranges.size());
		std::unordered_map<int64_t, size_t> width_freq;
		width_freq.reserve(std::min<size_t>(ranges.size(), 1u << 20));

		for (const auto& r : ranges) {
			min_l = std::min(min_l, r.l);
			max_l = std::max(max_l, r.l);
			min_u = std::min(min_u, r.u);
			max_u = std::max(max_u, r.u);

			if (r.l > r.u) {
				++invalid;
			}

			const int64_t w = static_cast<int64_t>(r.u) - static_cast<int64_t>(r.l);
			widths.push_back(w);
			min_w = std::min(min_w, w);
			max_w = std::max(max_w, w);
			++width_freq[w];

			if (r.l == r.u) {
				++point_ranges;
			}
		}

		const long double sum_w = std::accumulate(widths.begin(), widths.end(), 0.0L);
		const long double mean_w = sum_w / static_cast<long double>(widths.size());

		// Find top-5 most frequent widths.
		std::vector<std::pair<int64_t, size_t>> top_widths(width_freq.begin(), width_freq.end());
		std::sort(top_widths.begin(), top_widths.end(), [](const auto& a, const auto& b) {
			if (a.second != b.second) return a.second > b.second;
			return a.first < b.first;
		});
		if (top_widths.size() > 5) {
			top_widths.resize(5);
		}

		std::cout << "=== Range File Analysis ===\n";
		std::cout << "path: " << path << "\n";
		std::cout << "count: " << ranges.size() << "\n";
		std::cout << "L range: [" << min_l << ", " << max_l << "]\n";
		std::cout << "U range: [" << min_u << ", " << max_u << "]\n";
		std::cout << "invalid (L>U): " << invalid << "\n";
		std::cout << "point ranges (L==U): " << point_ranges << " ("
							<< std::fixed << std::setprecision(2)
							<< (100.0 * static_cast<double>(point_ranges) / static_cast<double>(ranges.size()))
							<< "%)\n";

		std::cout << "width stats:\n";
		std::cout << "  min=" << min_w
							<< " p50=" << Percentile(widths, 0.50)
							<< " p90=" << Percentile(widths, 0.90)
							<< " p99=" << Percentile(widths, 0.99)
							<< " max=" << max_w
							<< " mean=" << std::setprecision(3) << static_cast<double>(mean_w) << "\n";

		std::cout << "top widths (width: count):\n";
		for (const auto& [w, c] : top_widths) {
			std::cout << "  " << w << ": " << c << "\n";
		}

		std::cout << "sample first 10 ranges:\n";
		for (size_t i = 0; i < std::min<size_t>(10, ranges.size()); ++i) {
			std::cout << "  [" << ranges[i].l << ", " << ranges[i].u << "]\n";
		}

		std::cout << "=== Heuristic Inference ===\n";
		if (invalid > 0) {
			std::cout << "- file may not be generated by a proper [L,U] range generator (has L>U).\n";
		}

		if (point_ranges == ranges.size()) {
			std::cout << "- all ranges are point filters (likely generated by sampling attributes then using [att,att]).\n";
		} else if (point_ranges > ranges.size() * 8 / 10) {
			std::cout << "- mostly point filters, likely attribute-copy based with occasional perturbation.\n";
		}

		if (!top_widths.empty() && top_widths[0].second > ranges.size() * 8 / 10) {
			std::cout << "- dominant single width suggests fixed-window generation.\n";
		} else if (min_w >= 0 && max_w > min_w) {
			std::cout << "- variable non-negative widths suggest random L/R then swap (or sampled intervals).\n";
		}

		if (invalid == 0 && min_w >= 0) {
			std::cout << "- format matches test/data_loader.h::LoadRange (int32 L then int32 U, repeated).\n";
		}

		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
}

