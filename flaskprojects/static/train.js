new Vue({
    el: '#page2',
    delimiters: ["[[", "]]"],
    data: {
        file: [null, null],
        trainingProgress: [0, 0],
        taskId: [],
        modelName: ['', ''],
        trainingTime: [0, 0],
        tableData: [],
        isEditing: false,
        timer: [null, null],
        showDialog: false,
        selectedmodel: [],
        datasetUploadTimestamp: '',
        modelNamingTimestamp: '',
        trainingCompletionTimestamp: '',
        datasetName: '',
    },
    computed: {
        totalRows() {
            return this.tableData.length;
        },
        filteredData() {
            return this.tableData.filter(row => row.training_complete);
        }
    },
    methods: {
        handleFileChange(file) {
            this.file[0] = file.raw;
            this.datasetUploadTimestamp = new Date().toLocaleString();
            this.datasetName = file.name;
        },
        handleUploadSuccess() {
            this.$message.success('文件上传成功');
        },
        handleUploadError() {
            this.$message.error('文件上传失败');
        },
        startTraining() {
            if (this.file[0] === null) {
                this.$message.error('没有上传数据集文件');
                return;
            }
            if (this.modelName[0] === '') {
                this.$message.error('没有输入模型名称');
                return;
            }
            this.modelNamingTimestamp = new Date().toLocaleString();
            const formData = new FormData();
            formData.append('file', this.file[0]);
            formData.append('modelName', this.modelName[0]);
            axios.post('/train', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            })
                .then(response => {
                    let taskId = response.data.task_id;
                    this.timer[0] = setInterval(() => {
                        axios.get(`/train_progress/${taskId}`)
                            .then(progressResponse => {
                                this.trainingProgress[0] = progressResponse.data.progress;
                                this.$nextTick(() => {
                                    this.$forceUpdate();
                                });

                                if (this.trainingProgress[0] >= 100) {
                                    console.log(this.trainingProgress[0])
                                    this.trainingProgress[0] = 100;
                                    this.trainingCompletionTimestamp = new Date().toLocaleString();
                                    this.getModels();
                                    clearInterval(this.timer[0]);
                                }
                            })
                            .catch(progressError => {
                                console.error('Error fetching progress:', progressError.message);
                            });
                    }, 100);
                })

                .catch(error => {
                    this.$message.error('训练失败: ' + (error.message || ''));
                });
        },
        stripeRows({rowIndex}) {
            return rowIndex % 2 === 0 ? 'even-row' : 'odd-row';
        },
        getModels() {
            if (this.isEditing) {
                return;
            }
            axios.get('/models')
                .then(response => {
                    if (!response.data || response.data.length === 0) {
                        // this.$message.info('没有可用的模型');
                        return;
                    }
                    this.tableData = response.data.map(model => {
                        return {
                            ...model,
                        };
                    });
                })
            // .catch(error => {
            //     this.$message.error('获取模型失败: ' + (error.message || ''));
            // });
        },
        deleteModel(index, row) {
            axios.delete('/delete/' + row.id)
                .then(() => {
                    this.tableData.splice(index, 1);
                    this.$message.success('模型删除成功');
                    this.getModels();
                })
                .catch(() => {
                    this.$message.error('模型删除失败');
                });

        },
        editName(index, row) {
            this.isEditing = true;
            console.log(row.id)
            console.log(row.name)
            axios.post('/edit', {
                id: row.id,
                name: row.name
            }).then((response) => {
                this.getModels();
                if (response.data.success) {
                    this.$message.success('模型名称更改成功');
                } else {
                    this.$message.error(response.data.message);
                }
            }).catch((error) => {
                this.$message.error('请求失败');
            }).finally(() => {
                this.isEditing = false;
            });
        },
        downloadModel(id) {
            axios({
                url: '/download/' + id,
                method: 'GET',
                responseType: 'blob'
            }).then(response => {
                const url = window.URL.createObjectURL(new Blob([response.data]));
                const link = document.createElement('a');
                link.href = url;
                link.setAttribute('download', 'model.pth');
                document.body.appendChild(link);
                link.click();
            }).catch(error => {
                this.$message.error('下载失败: ' + (error.message || ''));
            });
        },
        resetModule() {
            this.file[0] = null;
            this.trainingProgress[0] = 0;
            this.modelName[0] = '';
            this.modelNamingTimestamp=null;
            this.datasetUploadTimestamp=null;
            this.trainingCompletionTimestamp=null;
            this.datasetName=null;
            if (this.timer[0]) {
                clearInterval(this.timer[0]);
                this.timer[0] = null;
            }
        },
        showDetails(row) {
            this.selectedmodel = row;
            this.showDialog = true;
        },
    },
    created() {
        // 轮询获取模型列表
        this.getModels();
        setInterval(this.getModels, 1000);
    }
});
