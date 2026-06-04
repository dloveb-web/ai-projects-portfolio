import { useState, useEffect } from 'react'
import { Row, Col, Card, Statistic, Table, Tag, Button, Modal, Form, Input, InputNumber, DatePicker, Select, message } from 'antd'
import { PlusOutlined, CalendarOutlined, AlertOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { productionApi } from '../../api/production'
import { materialApi } from '../../api/material'

export default function PMCDashboard() {
  const [orders, setOrders] = useState<any[]>([])
  const [materials, setMaterials] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [orderModalVisible, setOrderModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const [orderRes, matRes] = await Promise.all([
        productionApi.getOrders(),
        materialApi.getMaterials()
      ])
      setOrders(orderRes.data)
      setMaterials(matRes.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const ganttOption = {
    title: { text: '生产排程', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: ['周一', '周二', '周三', '周四', '周五', '周六', '周日'] },
    yAxis: { type: 'value' },
    series: [{
      type: 'bar',
      data: [2000, 2500, 2500, 2500, 2500, 1500, 0],
      itemStyle: { color: '#1890ff' }
    }]
  }

  const orderColumns = [
    { title: '订单编号', dataIndex: 'order_number', key: 'order_number' },
    { title: '产品名称', dataIndex: 'product_name', key: 'product_name' },
    { title: '数量', dataIndex: 'quantity', key: 'quantity' },
    { title: '优先级', dataIndex: 'priority', key: 'priority', render: (p: number) => {
      const colors: Record<number, string> = { 1: 'red', 2: 'orange', 3: 'blue' }
      const texts: Record<number, string> = { 1: '紧急', 2: '高', 3: '普通' }
      return <Tag color={colors[p]}>{texts[p]}</Tag>
    }},
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => {
      const colors: Record<string, string> = { pending: 'default', in_progress: 'processing', completed: 'success' }
      const texts: Record<string, string> = { pending: '待生产', in_progress: '生产中', completed: '已完成' }
      return <Tag color={colors[s]}>{texts[s]}</Tag>
    }},
    { title: '开始日期', dataIndex: 'start_date', key: 'start_date' },
    { title: '结束日期', dataIndex: 'end_date', key: 'end_date' },
  ]

  const handleSubmit = async (values: any) => {
    try {
      message.success('订单创建成功')
      setOrderModalVisible(false)
      fetchData()
    } catch (err) {
      message.error('创建失败')
    }
  }

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="待生产订单"
              value={5}
              prefix={<CalendarOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="生产中"
              value={8}
              prefix={<CalendarOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="本周产出"
              value={18000}
              prefix={<AlertOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="物料预警"
              value={3}
              prefix={<AlertOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card title="生产排程">
            <ReactECharts option={ganttOption} style={{ height: 300 }} />
          </Card>
        </Col>
      </Row>

      <Card 
        title="生产订单" 
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setOrderModalVisible(true)}>创建订单</Button>}
      >
        <Table 
          columns={orderColumns} 
          dataSource={orders}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="创建生产订单" open={orderModalVisible} onCancel={() => setOrderModalVisible(false)} onOk={() => form.submit()} width={600}>
        <Form form={form} onFinish={handleSubmit} layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="order_number" label="订单编号" rules={[{ required: true }]}>
                <Input placeholder="例如: PO2024001" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="product_name" label="产品名称" rules={[{ required: true }]}>
                <Input placeholder="例如: LCM-5.5" />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="quantity" label="数量" rules={[{ required: true }]}>
                <InputNumber style={{ width: '100%' }} min={1} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="priority" label="优先级" rules={[{ required: true }]}>
                <Select>
                  <Select.Option value={3}>普通</Select.Option>
                  <Select.Option value={2}>高</Select.Option>
                  <Select.Option value={1}>紧急</Select.Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="start_date" label="开始日期" rules={[{ required: true }]}>
                <DatePicker style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="end_date" label="结束日期" rules={[{ required: true }]}>
                <DatePicker style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </div>
  )
}
