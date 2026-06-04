import { useState, useEffect } from 'react'
import { Card, Table, Tag, Button, Modal, Form, Input, Select, message } from 'antd'
import { PlusOutlined, EyeOutlined } from '@ant-design/icons'
import { qualityApi } from '../../api/quality'

export default function InspectionPage() {
  const [inspections, setInspections] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const res = await qualityApi.getInspections()
      setInspections(res.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const columns = [
    { title: '产品编码', dataIndex: 'product_code', key: 'product_code' },
    { title: '检测类型', dataIndex: 'type', key: 'type', render: (type: string) => {
      const texts: Record<string, string> = { visual: '外观检测', electrical: '电性能检测', optical: '光学检测' }
      return texts[type] || type
    }},
    { title: '阶段', dataIndex: 'stage', key: 'stage', render: (stage: number) => {
      const stages = ['大板切割', 'COG/FOG', '贴合', '背光模组']
      return <Tag color="blue">{stages[stage - 1]}</Tag>
    }},
    { title: '检测结果', dataIndex: 'result', key: 'result', render: (result: string) => {
      const colors: Record<string, string> = { pass: 'green', fail: 'red', rework: 'orange' }
      const texts: Record<string, string> = { pass: '合格', fail: '不合格', rework: '需重工' }
      return <Tag color={colors[result]}>{texts[result]}</Tag>
    }},
    { title: '检测时间', dataIndex: 'created_at', key: 'created_at' },
    { title: '操作', key: 'action', render: (_: any, record: any) => (
      <Button size="small" icon={<EyeOutlined />}>查看</Button>
    )}
  ]

  const handleSubmit = async (values: any) => {
    try {
      message.success('检测记录创建成功')
      setCreateModalVisible(false)
      fetchData()
    } catch (err) {
      message.error('创建失败')
    }
  }

  return (
    <div>
      <Card 
        title="检测记录" 
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalVisible(true)}>添加记录</Button>}
      >
        <Table 
          columns={columns} 
          dataSource={inspections}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="添加检测记录" open={createModalVisible} onCancel={() => setCreateModalVisible(false)} onOk={() => form.submit()}>
        <Form form={form} onFinish={handleSubmit} layout="vertical">
          <Form.Item name="product_code" label="产品编码" rules={[{ required: true }]}>
            <Input placeholder="输入产品编码" />
          </Form.Item>
          <Form.Item name="type" label="检测类型" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="visual">外观检测</Select.Option>
              <Select.Option value="electrical">电性能检测</Select.Option>
              <Select.Option value="optical">光学检测</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="stage" label="阶段" rules={[{ required: true }]}>
            <Select>
              <Select.Option value={1}>大板切割</Select.Option>
              <Select.Option value={2}>COG/FOG</Select.Option>
              <Select.Option value={3}>贴合</Select.Option>
              <Select.Option value={4}>背光模组</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="result" label="检测结果" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="pass">合格</Select.Option>
              <Select.Option value="fail">不合格</Select.Option>
              <Select.Option value="rework">需重工</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="note" label="备注">
            <Input.TextArea rows={4} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
