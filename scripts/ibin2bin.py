import argparse
import os
import struct


def convert(input_path, output_path, dim):
	file_size = os.path.getsize(input_path)
	bytes_per_row = dim * 4
	if file_size % bytes_per_row != 0:
		raise ValueError(
			f"file size {file_size} is not divisible by dim*4={bytes_per_row}; "
			f"wrong --dim or corrupt input file"
		)
	count = file_size // bytes_per_row

	with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
		fout.write(struct.pack("<ii", count, dim))  # PipeANN .bin header: npts, dim (int32 each)
		while True:
			chunk = fin.read(1 << 20)
			if not chunk:
				break
			fout.write(chunk)

	return count


def main():
	parser = argparse.ArgumentParser(
		description="Convert a headerless raw int32 array (.ibin, e.g. WoW-style attribute file) "
		"into PipeANN's standard .bin format (8-byte <npts, dim> int32 header + raw payload)."
	)
	parser.add_argument("input", help="Input headerless raw int32 file (.ibin)")
	parser.add_argument("output", help="Output .bin file path (with npts/dim header)")
	parser.add_argument(
		"--dim",
		type=int,
		default=1,
		help="Number of int32 values per point (1 for a scalar range/label attribute)",
	)
	args = parser.parse_args()

	count = convert(args.input, args.output, args.dim)
	print(f"input={args.input}")
	print(f"output={args.output}")
	print(f"npts={count}, dim={args.dim}")


if __name__ == "__main__":
	main()
# python3 ibin2bin.py \
#     ${PATH_00019} \
#     ${PATH_00023} \
#     --dim 1