import { useState } from 'react'
import { Row, Col, Card, Statistic, Table, Tag, Button, Modal, Form, Input, message, Steps, Progress, Space } from 'antd'
import { 
  PlayCircleOutlined, 
  PauseCircleOutlined, 
  CheckCircleOutlined,
  SearchOutlined,
  QrcodeOutlined
} from '@ant-design/icons'

const { Step } = Steps

const mockWorkOrders = [
  { id: 1, work_order_number: 'WO-2024-001', product_code: 'LCM-IP5-001', stage: 1, current_product_code: 'LCM-IP5-001-A001', status: 'in_progress', quantity: 100, progress: 65 },
  { id: 2, work_order_number: 'WO-2024-002', product_code: 'LCM-IP5-002', stage: 2, current_product_code: 'LCM-IP5-002-B001', status: 'pending', quantity: 200, progress: 30 },
  { id: 3, work_order_number: 'WO-2024-003', product_code: 'LCM-IP5-003', stage: 3, current_product_code: 'LCM-IP5-003-C001', status: 'completed', quantity: 150, progress: 100 },
  { id: 4, work_order_number: 'WO-2024-004', product_code: 'LCM-IP5-004', stage: 4, current_product_code: 'LCM-IP5-004-D001', status: 'in_progress', quantity: 80, progress: 45 },
]

export default function OperatorDashboard() {
  const [workOrders] = useState<any[]>(mockWorkOrders)
  const [loading] = useState(false)
  const [traceModalVisible, setTraceModalVisible] = useState(false)
  const [traceForm] = Form.useForm()
  const [traceResult, setTraceResult] = useState<any>(null)

  const columns = [
    { title: '工单编号', dataIndex: 'work_order_number', key: 'work_order_number' },
    { title: '产品编码', dataIndex: 'product_code', key: 'product_code' },
    { title: '阶段', dataIndex: 'stage', key: 'stage', render: (stage: number) => {
      const stages = ['大板切割', 'COG/FOG', '贴合', '背光模组']
      return <Tag color="blue">{stages[stage - 1]}</Tag>
    }},
    { title: '当前编码', dataIndex: 'current_product_code', key: 'current_product_code' },
    { title: '数量', dataIndex: 'quantity', key: 'quantity' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (status: string) => {
      const colors: Record<string, string> = { pending: 'default', in_progress: 'processing', completed: 'success' }
      const texts: Record<string, string> = { pending: '待开始', in_progress: '进行中', completed: '已完成' }
      return <Tag color={colors[status]}>{texts[status]}</Tag>
    }},
    { title: '进度', dataIndex: 'progress', key: 'progress', render: (progress: number) => <Progress percent={progress} size="small" /> },
    { title: '操作', key: 'action', render: (_: any, record: any) => (
      <Space.Compact>
        <Button size="small" icon={<PlayCircleOutlined />}>开始</Button>
        <Button size="small" icon={<PauseCircleOutlined />}>暂停</Button>
        <Button size="small" type="primary" icon={<CheckCircleOutlined />}>完成</Button>
      </Space.Compact>
    )}
  ]

  const handleTrace = async (values: { code: string }) => {
    try {
      // 模拟产品追溯数据
      const mockTraceResult = {
        product_code: values.code,
        stage: 3,
        status: 'in_progress',
        stage_1_code: values.code + '-A001',
        stage_2_code: values.code + '-B001',
        stage_3_code: values.code + '-C001',
        stage_4_code: null
      }
      setTraceResult(mockTraceResult)
    } catch (err) {
      message.error('产品未找到')
    }
  }

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic title="待处理工单" value={5} valueStyle={{ color: '#faad14' }} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="进行中" value={8} valueStyle={{ color: '#1890ff' }} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="今日完成" value={15} valueStyle={{ color: '#52c41a' }} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="待返修" value={2} valueStyle={{ color: '#ff4d4f' }} />
          </Card>
        </Col>
      </Row>

      <Card 
        title="工单看板" 
        extra={<Button icon={<QrcodeOutlined />} onClick={() => setTraceModalVisible(true)}>产品追溯</Button>}
      >
        <Table 
          columns={columns} 
          dataSource={workOrders}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="产品追溯" open={traceModalVisible} onCancel={() => setTraceModalVisible(false)} footer={null}>
        <Form form={traceForm} onFinish={handleTrace} layout="vertical">
          <Form.Item name="code" label="产品编码" rules={[{ required: true }]}>
            <Input placeholder="输入8位/31位/71位码" prefix={<SearchOutlined />} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" block>追溯</Button>
          </Form.Item>
        </Form>
        {traceResult && (
          <Card title="追溯结果" size="small">
            <p><strong>当前编码:</strong> {traceResult.product_code}</p>
            <p><strong>阶段:</strong> {['大板切割', 'COG/FOG', '贴合', '背光模组'][traceResult.stage - 1]}</p>
            <p><strong>状态:</strong> {traceResult.status}</p>
            <div style={{ marginTop: 16 }}>
              <Steps direction="vertical" current={traceResult.stage}>
                <Step title="大板切割" description={traceResult.stage_1_code || '-'} />
                <Step title="COG/FOG" description={traceResult.stage_2_code || '-'} />
                <Step title="玻璃贴合" description={traceResult.stage_3_code || '-'} />
                <Step title="背光模组" description={traceResult.stage_4_code || '-'} />
              </Steps>
            </div>
          </Card>
        )}
      </Modal>
    </div>
  )
}
