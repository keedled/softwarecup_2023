new Vue({
    el: '#page4',
    delimiters: ["[[", "]]"],
    data: {
        total_data: {},
        daily_data: {},
        today_data: {},
        myChart: null,
    },
    mounted() {
        this.myChart = echarts.init(document.getElementById('page4-echarts'));
        this.fetchData(); // Immediately invoke once at start
        setInterval(() => {
            this.fetchData(); // This will execute every 4 seconds
        }, 4000);
    },
    methods: {
        fetchData() {
            axios.get('/five_day_data')
                .then(response => {
                    this.total_data = {
                        users: response.data.total_users,
                        models: response.data.total_models,
                        predictions: response.data.total_predictions,
                    };

                    this.daily_data = response.data.daily_data;

                    let dates = Object.keys(this.daily_data);
                    let new_users = dates.map(date => this.daily_data[date].new_users);
                    let new_models = dates.map(date => this.daily_data[date].new_models);
                    let new_predictions = dates.map(date => this.daily_data[date].new_predictions);

                    let today = new Date();
                    today = today.getFullYear() + '-' + String(today.getMonth() + 1).padStart(2, '0') + '-' + String(today.getDate()).padStart(2, '0');  // get today's date in yyyy-mm-dd format
                    let today_data = this.daily_data[today];

                    this.today_data = {
                        users: today_data ? today_data.new_users : 0,
                        models: today_data ? today_data.new_models : 0,
                        predictions: today_data ? today_data.new_predictions : 0,
                    };
                    let option = {
                        title: {
                            text: '近五日数据'
                        },
                        tooltip: {},
                        legend: {
                            data: ['新增用户数', '训练模型数', '模型预测数']
                        },
                        xAxis: {
                            data: dates
                        },
                        yAxis: {},
                        series: [
                            {
                                name: '新增用户数',
                                type: 'bar',
                                data: new_users,
                                itemStyle: {
                                    color: 'blue'
                                }
                            },
                            {
                                name: '训练模型数',
                                type: 'bar',
                                data: new_models,
                                itemStyle: {
                                    color: 'red'
                                }
                            },
                            {
                                name: '模型预测数',
                                type: 'bar',
                                data: new_predictions,
                                itemStyle: {
                                    color: 'green'
                                }
                            }
                        ]
                    };

                    this.myChart.setOption(option);
                })
                .catch(error => {
                    console.log(error);
                });
        }
    }
});
