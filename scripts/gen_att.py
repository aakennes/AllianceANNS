import argparse
import os
import random
import struct


def get_fvecs_count(path):
	with open(path, "rb") as f:
		head = f.read(4)
		if len(head) != 4:
			raise ValueError("fvecs file is empty or broken")
		dim = struct.unpack("<i", head)[0]
		if dim <= 0:
			raise ValueError(f"invalid dimension in fvecs header: {dim}")

	file_size = os.path.getsize(path)
	bytes_per_vec = 4 + dim * 4
	if file_size % bytes_per_vec != 0:
		raise ValueError(
			f"file size {file_size} is not divisible by vector stride {bytes_per_vec}; not a valid fvecs file"
		)
	return file_size // bytes_per_vec, dim


def write_unique_attributes(output_path, count, seed=None):
	rng = random.Random(seed)
	attributes = list(range(0, count))
	rng.shuffle(attributes)

	with open(output_path, "wb") as out:
		for value in attributes:
			out.write(struct.pack("<i", value))


def main():
	parser = argparse.ArgumentParser(
		description="Generate raw int32 attribute file (.ibin) for WoW build from fvecs vector count."
	)
	parser.add_argument(
		"--basevec",
		default="data/Sift1m/sift_base.fvecs",
		help="Input fvecs path used to infer vector count",
	)
	parser.add_argument(
		"--output",
		default="data/Sift1m/sift_base.attribute.ibin",
		help="Output attribute ibin path (raw int32 array, no header)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducible shuffling",
	)
	args = parser.parse_args()

	count, dim = get_fvecs_count(args.basevec)
	write_unique_attributes(args.output, count, args.seed)
	print(f"basevec={args.basevec}")
	print(f"dimension={dim}, vectors={count}")
	print(f"output={args.output}, value_range=[0, {count-1}], unique=true, seed={args.seed}")


if __name__ == "__main__":
	main()
