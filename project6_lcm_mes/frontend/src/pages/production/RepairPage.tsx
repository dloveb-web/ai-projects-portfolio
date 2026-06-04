import { useState, useEffect } from 'react'
import { Card, Table, Tag, Button, Modal, Form, Input, Select, message, Space } from 'antd'
import { ToolOutlined, CheckOutlined, CloseOutlined } from '@ant-design/icons'
import { qualityApi } from '../../api/quality'

export default function RepairPage() {
  const [defects, setDefects] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [repairModalVisible, setRepairModalVisible] = useState(false)
  const [currentDefect, setCurrentDefect] = useState<any>(null)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const res = await qualityApi.getDefects()
      setDefects(res.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const columns = [
    { title: '产品编码', dataIndex: 'product_code', key: 'product_code' },
    { title: '阶段', dataIndex: 'stage', key: 'stage', render: (stage: number) => {
      const stages = ['大板切割', 'COG/FOG', '贴合', '背光模组']
      return <Tag color="blue">{stages[stage - 1]}</Tag>
    }},
    { title: '缺陷类型', dataIndex: 'defect_type', key: 'defect_type', render: (type: string) => {
      const colors: Record<string, string> = { scratch: 'red', spot: 'orange', crack: 'red', misalignment: 'blue' }
      const texts: Record<string, string> = { scratch: '划伤', spot: '斑点', crack: '裂纹', misalignment: '偏移' }
      return <Tag color={colors[type]}>{texts[type] || type}</Tag>
    }},
    { title: '严重程度', dataIndex: 'severity', key: 'severity', render: (s: string) => {
      const colors: Record<string, string> = { critical: 'red', major: 'orange', minor: 'blue' }
      const texts: Record<string, string> = { critical: '严重', major: '主要', minor: '轻微' }
      return <Tag color={colors[s]}>{texts[s]}</Tag>
    }},
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => {
      const colors: Record<string, string> = { pending: 'orange', repairing: 'processing', repaired: 'green', scrapped: 'red' }
      const texts: Record<string, string> = { pending: '待处理', repairing: '返修中', repaired: '已修复', scrapped: '报废' }
      return <Tag color={colors[s]}>{texts[s]}</Tag>
    }},
    { title: '返修次数', dataIndex: 'repair_count', key: 'repair_count' },
    { title: '操作', key: 'action', render: (_: any, record: any) => (
      <Space.Compact>
        {record.status === 'pending' && (
          <Button size="small" icon={<ToolOutlined />} onClick={() => handleRepair(record)}>返修</Button>
        )}
        {record.status === 'repairing' && (
          <>
            <Button size="small" type="primary" icon={<CheckOutlined />} onClick={() => handleComplete(record)}>完成</Button>
            <Button size="small" danger icon={<CloseOutlined />} onClick={() => handleScrap(record)}>报废</Button>
          </>
        )}
      </Space.Compact>
    )}
  ]

  const handleRepair = (record: any) => {
    setCurrentDefect(record)
    setRepairModalVisible(true)
  }

  const handleComplete = (record: any) => {
    message.success('返修完成')
    fetchData()
  }

  const handleScrap = (record: any) => {
    message.warning('产品已报废')
    fetchData()
  }

  const handleSubmit = async (values: any) => {
    try {
      message.success('返修记录已创建')
      setRepairModalVisible(false)
      fetchData()
    } catch (err) {
      message.error('创建失败')
    }
  }

  return (
    <div>
      <Card title="不良品维修">
        <Table 
          columns={columns} 
          dataSource={defects}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="返修操作" open={repairModalVisible} onCancel={() => setRepairModalVisible(false)} onOk={() => form.submit()}>
        <Form form={form} onFinish={handleSubmit} layout="vertical">
          <Form.Item label="产品编码" name="product_code">
            <Input disabled />
          </Form.Item>
          <Form.Item label="返修方案" name="solution" rules={[{ required: true }]}>
            <Select style={{ width: '100%' }}>
              <Select.Option value="clean">清洁</Select.Option>
              <Select.Option value="rework">重工</Select.Option>
              <Select.Option value="replace">更换</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item label="备注" name="note">
            <Input.TextArea rows={4} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
