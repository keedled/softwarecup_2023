new Vue({
    el: '#page3',
    delimiters: ["[[", "]]"],
    data: {
        testFile: null,
        selectedModelId: null,
        models: [],
        tableData2: []
    },
    methods: {
        handleTestFileChange(file) {
            this.testFile = file.raw;
        },
        handleUploadSuccess() {
            this.$message.success('文件上传成功');
        },
        handleUploadError() {
            this.$message.error('文件上传失败');
        },
        startTesting() {
            if (this.testFile === null) {
                this.$message.error('没有上传测试数据集文件');
                return;
            }
            if (this.selectedModelId === null) {
                this.$message.error('没有选择模型');
                return;
            }
            console.log(this.selectedModelId)
            const formData = new FormData();
            formData.append('test_data', this.testFile);
            formData.append('model_id', this.selectedModelId);
            axios.post('/test', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            })
                .then(response => {

                })
                .catch(error => {
                    this.$message.error('测试失败: ' + (error.message || ''));
                });
        },
        getModels() {
            axios.get('/models2')
                .then(response => {
                    this.models = response.data;
                })
                .catch(error => {
                    this.$message.error('获取模型失败: ' + (error.message || ''));
                });
        },
        fetchPredictions() {
            axios.get('/predictions').then(response => {
                this.tableData2 = response.data.map(prediction => {
                    var imagePath = prediction.result_image_url.replace('results/', '');
                    var imageSrc = '/results/' + imagePath;
                    return {
                        id: prediction.id,
                        imageSrc: imageSrc,
                        jsonurl:prediction.result_json_url,
                        data: [
                            prediction.class0_count,
                            prediction.class1_count,
                            prediction.class2_count,
                            prediction.class3_count,
                            prediction.class4_count,
                            prediction.class5_count
                        ]
                    };
                });
            })
                .catch(error => {
                    console.error(error);
                });
        },
        deleteResult(id) {
            axios.delete(`/delete_result/${id}`)
                .then(response => {
                    if (response.data.success) {
                        this.tableData2 = this.tableData2.filter(prediction => prediction.id !== id);
                    } else {
                        console.log("Delete failed");
                    }
                })
                .catch(error => {
                    console.log(error);
                });
        },
        resetPredictionModule() {
        this.selectedModelId = null;
        this.testFile = null;
    },
    },
    created() {
        this.getModels();  // 页面加载后立即获取模型列表
        setInterval(this.getModels, 3000);
        this.fetchPredictions();
        this.interval = setInterval(() => {
            this.fetchPredictions();
        }, 2000);
    },
    beforeDestroy() {
        clearInterval(this.interval);
    }
});
