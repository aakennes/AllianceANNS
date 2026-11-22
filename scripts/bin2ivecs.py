import struct


def convert_bin_to_ivecs(input_file, output_file):
	with open(input_file, 'rb') as f:

		num_vectors = struct.unpack('<I', f.read(4))[0]  # 小端无符号int
		dimension = struct.unpack('<I', f.read(4))[0]

		# 计算每个向量的大小（字节数）
		vector_size = dimension * 4  # 每个int32占4字节

		# 逐个处理每个向量
		with open(output_file, 'wb') as out:
			for _ in range(num_vectors):
				# 读取单个向量数据
				vector_data = f.read(vector_size)

				# 写入维度头（int32）和向量数据
				out.write(struct.pack('<I', dimension))
				out.write(vector_data)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    convert_bin_to_ivecs(args.input, args.output)
