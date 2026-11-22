import argparse
import struct


def convert(input_path, output_path, k=None, pad_id=0xFFFFFFFF):
	# AllianceANNS gengt format (see src/tools/dataset/gen_ground_truth.cpp): a
	# sequence of per-query records, each "int32 k_i" followed by "k_i x
	# uint32 id". k_i can vary per query (attribute-filtered brute force may
	# return fewer than the requested k).
	records = []
	with open(input_path, "rb") as fin:
		while True:
			head = fin.read(4)
			if len(head) == 0:
				break
			if len(head) != 4:
				raise ValueError("truncated k field at record " + str(len(records)))
			(row_k,) = struct.unpack("<i", head)
			ids_bytes = fin.read(row_k * 4)
			if len(ids_bytes) != row_k * 4:
				raise ValueError("truncated ids block at record " + str(len(records)))
			ids = list(struct.unpack("<%dI" % row_k, ids_bytes))
			records.append(ids)

	npts = len(records)
	dim = k if k is not None else max((len(r) for r in records), default=0)

	# PipeANN load_truthset (include/utils.h) expects a fixed dim for every
	# query: <int32 npts><int32 dim><uint32 ids[npts*dim]> (ids-only /
	# "truthset_type == 2", since AllianceANNS gt carries no distances).
	# Row-major: all npts*dim ids are written as one contiguous block, not
	# interleaved per query.
	with open(output_path, "wb") as fout:
		fout.write(struct.pack("<ii", npts, dim))
		for ids in records:
			if len(ids) > dim:
				raise ValueError(
					f"record has {len(ids)} ids, exceeds target dim={dim}; pass --k >= max row length"
				)
			row = ids + [pad_id] * (dim - len(ids))
			fout.write(struct.pack("<%dI" % dim, *row))

	return npts, dim


def main():
	parser = argparse.ArgumentParser(
		description="Convert AllianceANNS ground-truth output (alliance_gengt, per-query "
		"variable-length [int32 k][uint32 ids...] records, no distances) into PipeANN's "
		"load_truthset .bin format (ids-only: 8-byte <npts,dim> int32 header + npts*dim "
		"uint32 ids, row-major)."
	)
	parser.add_argument("input", help="Input AllianceANNS ground-truth file (from alliance_gengt --gt_file)")
	parser.add_argument("output", help="Output PipeANN-compatible .bin ground-truth file")
	parser.add_argument(
		"--k",
		type=int,
		default=None,
		help="Fixed dim (top-k) for the output. Defaults to the longest per-query record "
		"found in the input. Rows shorter than this are padded with --pad_id.",
	)
	parser.add_argument(
		"--pad_id",
		type=lambda x: int(x, 0),
		default=0xFFFFFFFF,
		help="uint32 sentinel used to pad queries with fewer than --k ground-truth ids "
		"(default 0xFFFFFFFF, matching PipeANN's own missing-neighbor convention)",
	)
	args = parser.parse_args()

	npts, dim = convert(args.input, args.output, k=args.k, pad_id=args.pad_id)
	print(f"input={args.input}")
	print(f"output={args.output}")
	print(f"npts={npts}, dim={dim}")


if __name__ == "__main__":
	main()
# python3 gt_to_pipeann_bin.py \
#     ${PATH_00021} \
#     ${PATH_00022} \
#     --k 25
