new Vue({
    el: '#page5',
    delimiters: ["[[", "]]"],
    data: {
        form: {
            content: ''
        },
        keywords: '',
        tableData: [],
        currentTableData: [],
        currentPage: 1,
        drawer: false,
        userDetail: {},
        userId: null,
        mainBarChartData: {},
        sortedBarChartData: {},
    },
    methods: {
        formatRole(row, column, cellValue) {
            switch (cellValue) {
                case 1:
                    return '用户';
                case 2:
                    return '管理员';
                case 3:
                    return '超级管理员';
                default:
                    return '未知';
            }
        },
        submitForm(formName) {
            let formData = this.$refs[formName].model;
            axios.post('/announcement', formData).then(response => {
                this.form.content = '';
                this.$message({
                    message: '公告发布成功！',
                    type: 'success'
                });
            }).catch(error => {
                console.error('Error publishing announcement:', error);
            });
        },
        fetchUsers() {
            axios.get('/users').then(response => {
                this.tableData = response.data;
                this.handlePageChange(this.currentPage);
            }).catch(error => {
                console.error('Error fetching users:', error);
            });
        },
        search() {
            this.currentTableData = this.tableData.filter(user => user.nickname.includes(this.keywords) || user.username.includes(this.keywords));
        },
        reset() {
            this.keywords = '';
            this.handlePageChange(this.currentPage);
        },
        formatTime(gmtTimeString) {
            const months = {
                'Jan': '01',
                'Feb': '02',
                'Mar': '03',
                'Apr': '04',
                'May': '05',
                'Jun': '06',
                'Jul': '07',
                'Aug': '08',
                'Sep': '09',
                'Oct': '10',
                'Nov': '11',
                'Dec': '12'
            };
            const parts = gmtTimeString.split(' ');
            const day = parts[1];
            const month = months[parts[2]];
            const year = parts[3];
            const timeParts = parts[4].split(':');
            const hour = timeParts[0];
            const minute = timeParts[1];
            const second = timeParts[2];
            return `${year}年${month}月${day}日 ${hour}:${minute}:${second}`;
        },
        showdetails(rowData) {
            this.userId = rowData.id;
            axios.get(`/get_user_details/${rowData.id}`)
                .then(response => {
                    if (response.data) {
                        this.userDetail.model_count = response.data.model_count;
                        this.userDetail.prediction_count = response.data.prediction_count;
                        this.userDetail.login_latest_time = this.formatTime(response.data.login_latest_time);
                        this.userDetail.model_latest_time = this.formatTime(response.data.model_latest_time);
                        this.userDetail.prediction_latest_time = this.formatTime(response.data.prediction_latest_time);

                        this.drawer = true;
                    } else {
                        console.error("Received unexpected data format from the backend.");
                    }
                })
                .catch(error => {
                    console.error("Error fetching user details:", error.message);
                });
        },
        banUser(row) {
            axios.post(`/ban/${row.id}`).then(response => {
                if (response.data.message === "用户已被封号") {
                    this.$message.success('用户已被封号');
                    row.is_banned = 0;
                } else {
                    this.$message.error(response.data.message);
                }
            }).catch(error => {
                console.error('Error banning user:', error);
                this.$message.error('操作失败');
            });
        },
        unbanUser(row) {
            axios.post(`/unban/${row.id}`).then(response => {
                if (response.data.message === "用户已解除封号") {
                    this.$message.success('用户已解除封号');
                    row.is_banned = 1;
                } else {
                    this.$message.error(response.data.message);
                }
            }).catch(error => {
                console.error('Error unbanning user:', error);
                this.$message.error('操作失败');
            });
        },
        handlePageChange(page) {
            let start = (page - 1) * 7;
            let end = start + 7;
            this.currentTableData = this.tableData.slice(start, end);
        },
        convertToTimeRange(timePoint) {
            switch (timePoint) {
                case '03:00':
                    return '00:00-03:00';
                case '06:00':
                    return '03:00-06:00';
                case '09:00':
                    return '06:00-09:00';
                case '12:00':
                    return '09:00-12:00';
                case '15:00':
                    return '12:00-15:00';
                case '18:00':
                    return '15:00-18:00';
                case '21:00':
                    return '18:00-21:00';
                case '24:00':
                    return '21:00-24:00';
                default:
                    return timePoint;
            }
        },

        loadLastSevenDaysData() {
            axios.get(`/login_data/${this.userId}`).then(response => {
                this.mainBarChartData = response.data;
                let mainBarChart = echarts.init(this.$refs.mainBarChart);
                mainBarChart.setOption({
                    title: {
                        text: '近七天的登录数据'
                    },
                    xAxis: {
                        data: Object.keys(this.mainBarChartData),
                        axisLabel: {
                            // rotate: 45
                            interval: 0
                        }
                    },
                    yAxis: {},
                    series: [{
                        name: '登录次数',
                        type: 'bar',
                        data: Object.values(this.mainBarChartData)
                    }]
                });
                mainBarChart.on('click', (params) => {
                    console.log("Clicked date:", params.name);
                    this.loadLoginDetailForDate(params.name);
                });
            });
        },
        loadLoginDetailForDate(date) {
            console.log(date)
            axios.get(`/login_data_detail/${this.userId}/${date}`).then(response => {
                this.sortedBarChartData = response.data;
                let sortedData = Object.entries(this.sortedBarChartData).sort((a, b) => b[1] - a[1]);
                let yAxisData = sortedData.map(item => this.convertToTimeRange(item[0])).reverse(); // 反转数组
                let seriesData = sortedData.map(item => item[1]).reverse();  // 反转数组
                let sortedBarChart = echarts.init(this.$refs.sortedBarChart);
                sortedBarChart.setOption({
                    title: {
                        text: `登录详情 - ${date}`
                    },
                    xAxis: {
                        type: 'value'
                    },
                    yAxis: {
                        type: 'category',
                        data: yAxisData,
                        axisLabel: {
                            interval: 0,
                            rotate: 45
                        }
                    },
                    series: [{
                        name: '登录次数',
                        type: 'bar',
                        data: seriesData
                    }]
                });
            });
        },

        handleDrawerOpened() {
            this.loadLastSevenDaysData();
        },
        shouldPollData() {
        if (!this.keywords) {
            this.fetchUsers();
        }
    }
    },
    mounted() {
        this.fetchUsers();
        setInterval(this.shouldPollData, 3000);
    }
});
