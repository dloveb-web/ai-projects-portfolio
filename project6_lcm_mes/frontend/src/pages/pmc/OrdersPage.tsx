import { useState, useEffect } from 'react'
import { Card, Table, Tag, Button, Modal, Form, Input, Select, InputNumber, DatePicker, message, Space } from 'antd'
import { PlusOutlined, ReloadOutlined } from '@ant-design/icons'
import { productionApi } from '../../api/production'

export default function OrdersPage() {
  const [orders, setOrders] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const res = await productionApi.getOrders()
      setOrders(res.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const columns = [
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
    { title: '操作', key: 'action', render: (_: any, record: any) => (
      <Button size="small" onClick={() => handleEdit(record)}>编辑</Button>
    )}
  ]

  const handleCreate = () => {
    setCreateModalVisible(true)
  }

  const handleEdit = (record: any) => {
    form.setFieldsValue(record)
    setCreateModalVisible(true)
  }

  const handleSubmit = async (values: any) => {
    try {
      message.success('订单保存成功')
      setCreateModalVisible(false)
      fetchData()
    } catch (err) {
      message.error('保存失败')
    }
  }

  return (
    <div>
      <Card 
        title="订单管理" 
        extra={
          <Space.Compact>
            <Button icon={<ReloadOutlined />} onClick={fetchData}>刷新</Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={handleCreate}>创建订单</Button>
          </Space.Compact>
        }
      >
        <Table 
          columns={columns} 
          dataSource={orders}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="订单" open={createModalVisible} onCancel={() => setCreateModalVisible(false)} onOk={() => form.submit()} width={600}>
        <Form form={form} onFinish={handleSubmit} layout="vertical">
          <Form.Item name="order_number" label="订单编号" rules={[{ required: true }]}>
            <Input placeholder="例如: PO2024001" />
          </Form.Item>
          <Form.Item name="product_name" label="产品名称" rules={[{ required: true }]}>
            <Input placeholder="例如: LCM-5.5" />
          </Form.Item>
          <Form.Item name="quantity" label="数量" rules={[{ required: true }]}>
            <InputNumber style={{ width: '100%' }} min={1} />
          </Form.Item>
          <Form.Item name="priority" label="优先级" rules={[{ required: true }]}>
            <Select>
              <Select.Option value={3}>普通</Select.Option>
              <Select.Option value={2}>高</Select.Option>
              <Select.Option value={1}>紧急</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="start_date" label="开始日期">
            <DatePicker style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="end_date" label="结束日期">
            <DatePicker style={{ width: '100%' }} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
