import { useState, useEffect } from 'react'
import { Card, Table, Tag, Button, Modal, Form, Select, InputNumber, message, Space } from 'antd'
import { CheckCircleOutlined, PlayCircleOutlined } from '@ant-design/icons'
import { productionApi } from '../../api/production'

export default function WorkOrderPage() {
  const [workOrders, setWorkOrders] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [reportModalVisible, setReportModalVisible] = useState(false)
  const [currentOrder, setCurrentOrder] = useState<any>(null)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const res = await productionApi.getWorkOrders()
      setWorkOrders(res.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const columns = [
    { title: '工单编号', dataIndex: 'work_order_number', key: 'work_order_number' },
    { title: '产品编码', dataIndex: 'product_code', key: 'product_code' },
    { title: '阶段', dataIndex: 'stage', key: 'stage', render: (stage: number) => {
      const stages = ['大板切割', 'COG/FOG', '贴合', '背光模组']
      return <Tag color="blue">{stages[stage - 1]}</Tag>
    }},
    { title: '数量', dataIndex: 'quantity', key: 'quantity' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (status: string) => {
      const colors: Record<string, string> = { pending: 'default', in_progress: 'processing', completed: 'success' }
      const texts: Record<string, string> = { pending: '待开始', in_progress: '进行中', completed: '已完成' }
      return <Tag color={colors[status]}>{texts[status]}</Tag>
    }},
    { title: '开始时间', dataIndex: 'start_time', key: 'start_time' },
    { title: '操作', key: 'action', render: (_: any, record: any) => (
      <Space.Compact>
        {record.status === 'pending' && (
          <Button size="small" icon={<PlayCircleOutlined />} onClick={() => handleStart(record)}>开始</Button>
        )}
        {record.status === 'in_progress' && (
          <Button size="small" type="primary" icon={<CheckCircleOutlined />} onClick={() => handleReport(record)}>报工</Button>
        )}
      </Space.Compact>
    )}
  ]

  const handleStart = (record: any) => {
    message.success('工单已开始')
    fetchData()
  }

  const handleReport = (record: any) => {
    setCurrentOrder(record)
    setReportModalVisible(true)
  }

  const handleSubmit = async (values: any) => {
    try {
      message.success('报工成功')
      setReportModalVisible(false)
      fetchData()
    } catch (err) {
      message.error('报工失败')
    }
  }

  return (
    <div>
      <Card title="工单操作">
        <Table 
          columns={columns} 
          dataSource={workOrders}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="生产报工" open={reportModalVisible} onCancel={() => setReportModalVisible(false)} onOk={() => form.submit()}>
        <Form form={form} onFinish={handleSubmit} layout="vertical" initialValues={currentOrder}>
          <Form.Item label="工单编号" name="work_order_number">
            <InputNumber disabled style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item label="合格数量" name="good_quantity" rules={[{ required: true }]}>
            <InputNumber style={{ width: '100%' }} min={0} />
          </Form.Item>
          <Form.Item label="不良数量" name="defect_quantity" rules={[{ required: true }]}>
            <InputNumber style={{ width: '100%' }} min={0} />
          </Form.Item>
          <Form.Item label="Lot号" name="lot_number">
            <Select style={{ width: '100%' }} placeholder="选择Lot号">
              <Select.Option value="LOT001">LOT001</Select.Option>
              <Select.Option value="LOT002">LOT002</Select.Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
