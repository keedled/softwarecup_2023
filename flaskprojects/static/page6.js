new Vue({
    el: '#page6',
    delimiters: ["[[", "]]"],
    data: {
        form: {
            content: ''
        },
        keywords: '',
        currentTableData: [],
        currentPage: 1,
        tableData: [],
    },
    methods: {
        formatBannedStatus(row) {
            return row.is_banned === 1 ? '正常' : '被封号';
        },

        statusStyle(isBanned) {
            return {
                color: isBanned === 1 ? 'green' : 'red'
            };
        },

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
        fetchUsers() {
            axios.get('/users').then(response => {
                this.tableData = response.data;
                this.handlePageChange(this.currentPage);  // 初始化当前页面数据
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
        handlePageChange(page) {
            let start = (page - 1) * 7;
            let end = start + 7;
            this.currentTableData = this.tableData.slice(start, end);
        },
        setAdmin(row) {
            axios.post('/set-admin', {userId: row.id})
                .then(response => {
                    if (response.data.success) {
                        row.role = 2;
                    }
                })
                .catch(error => {
                    console.error("Error setting as admin:", error);
                });
        },
        removeAdmin(row) {
            axios.post('/remove-admin', {userId: row.id})
                .then(response => {
                    if (response.data.success) {
                        row.role = 1;
                    }
                })
                .catch(error => {
                    console.error("Error removing as admin:", error);
                });
        },
        shouldPollData() {
            if (!this.keywords) {
                this.fetchUsers();
            }
        },
    },
    mounted() {
        this.fetchUsers();
        if (keywords !== '') {
            setInterval(this.fetchUsers, 3000);
        }
        setInterval(this.shouldPollData, 3000);
    }
});
