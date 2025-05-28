#代码中的路径问题还请酌情修改，有问题请及时联系我们

#新建B_test 测试B榜
cd /root
mkdir B_test
cd B_test

#B榜的数据集
ossutil cp oss://tianchi-race-prod-sh/file/race/documents/prod/532324/1514/public/B.zip ./B.zip -i STS.NVAbAFDABj6o4xQoLejrYPyH7 -k 9PJ58sVN46w1ajY4k8N8gKKnqAsX2sz6B3mCE7toUGwC --endpoint=oss-cn-shanghai.aliyuncs.com --sts-token=CAISswN1q6Ft5B2yfSjIr5X0KfvyqZ5j3fSENl7gi0wwZv11v7zj1Tz2IHhPfHlpAe0Zs/Q/nWpW6PYclrhvQKhJTFDNacJ62ckMq1j7P9IAATF7a+ZW5qe+EE2/VjQ6ta27OpcyJbGwU/OpbE++2U0X6LDmdDKkckW4OJmS8/BOZcgWWQ/KClgjA8xNdCRvtOgQN3baKYy0UHjQj3HXEVBjtydllGp78t7f+MCH7QfEh1CI940uro/qcJ+/dJsubtUtT9a82ud2d+/b2SVdrgBQ86szl6wD9zbDs5aHClJcpBmBOPfR/9tzN0hhfK82XPMf9qOmyboh57WP0JzqwRJMNqZ/FTbeXMKCuJKYQbr5Z4lhLO6qZSqQgo21W8Or419+UxUyLxhXftctEHh0BCE3RyvSQq3dowiROlf6E/XajfFmiMArkQW256SWJlGJSLWYlCoRJpYnYl9tKgYS2mXtYn7OqvrqmPoN7d+3OmYTBHg2wcuA1Ybix6jMYm55lUlfdiF7y0EO6PQer6g8pa43TrooHQUkAao5NfzQpGnqOkIrv9Qy8u3ZXbXbjhvttVpji7CuYpgagAFRm3bJpQR6oVkb10l8q6x+0ijnh5KZIsS2O3g4nUSQcQ0EVBG5DpSLXgjMOrPir9ls/e12tSX2Uq5DD5+ZWDVALdQyeD3XoUy3lEwrp87GCGMPFeAJMqUukWLVXAaR0B+zghquAOFISa/gBuAyUgr+TEZCws35S62KT9mhfAySCyAA

unzip B.zip


ossutil cp oss://tianchi-race-prod-sh/file/race/documents/prod/532324/1514/public/label_B.zip ./label_B.zip -i STS.NWw5DPhmU4xFGYNsvgKQJc9fe -k 4fst2C1RvHyfzpV9THyQPcfEHBu9BetZtbCz3p65LeWn --endpoint=oss-cn-shanghai.aliyuncs.com --sts-token=CAISuQN1q6Ft5B2yfSjIr5TCfv7khbJ0g7qtRX//l3YyR95mjPzNhzz2IHhPfHlpAe0Zs/Q/nWpW6PYclrhvQKhJTFDNacJ62ckMq1j7P9I7JmB7a+ZW5qe+EE2/VjQ8ta27Opc4JbGwU/OpbE++2U0X6LDmdDKkckW4OJmS8/BOZcgWWQ/KClgjA8xNdCRvtOgQN3baKYyyUHjQj3HXEVBjtydllGp78t7f+MCH7QfEh1CI+Y0kro/qcJ+/dJsubtUtT9a82ud2d+/b2SVdrgBQ86szl6wD9zbDs5aHClJcpBmBOPfR/9tzN0hhfK82XPMf9qOmyboh57WP0JzqwRJMNqZRWi7SQLeKhtnFAKGLTo9lLO+raiuVi4jeaMeo414eDChFZF8QSb0IMWRtDBEgcDbeJ5K89UrCCgXZEPDeiPFviccqkgW4pIDUeQjRWcuF0C8eMZ89Kk8hKxcN0HCkb7cCdAVAY4Kv3kAzpiwZhnl83kCI1WW6PkUIpvqn42ZRy+z4kBS+eFdSL3NXUPv8la48x08AZXqZ5dygxLnaCiIbb5ionoejx/3byvuxpM/3B0qusO2JNQsyGVgagAE+1nJe9U69JhnG1G2Fy8OzkkRb6d0LFAUVZe78/th+eFNL4M0Urxb8yOiZj80l+x2PaBhi3kGHF1CieRjAGcMkqyZ7NjX6GIrz727enynCspky0cKI8cg3VOl5geGDdbbkKwG5VGylziMngtg6qiZYVDWWoMX7CfOVKGot64E+jCAA

unzip label_B.zip


# 使用多标签分类模型qwen2.5-vl-7b-sft模型进行初次推理，再使用高风险的分类模型qwen2.5-vl-3b-sft-high-risk进行校正合并得到检测结果b-submit

#首先从魔塔社区下载我们训练好的多标签分类模型和高风险模型
# 多标签分类模型
modelscope download --model llyllylly/Qwen2.5-VL-7B-Instruct-fine-tuning4-lora-aug-5.4529-a1-5.1567 --local_dir /root/autodl-tmp/project/model/

# 高风险模型
modelscope download --model llyllylly/Qwen2.5-VL-3B-Instruct-aug-12e --local_dir /root/autodl-tmp/project/model/


# 模型推理
python /root/autodl-tmp/project/code/qwen2.5-vl-7b-sft-eva.py   #得到多标签分类模型预测结果
python /root/autodl-tmp/project/code/qwen2.5-vl-3b-sft-high-risk-eva.py      #得到高风险预测模型结果
python /root/autodl-tmp/project/code/3b_merge_7b_high_ricks.py           #合并两个模型预测结果得到最终的结果文件
