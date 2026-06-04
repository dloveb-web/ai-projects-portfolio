import { useState, useEffect } from 'react'
import { Card, Table, Tag, Button, Modal, Form, Input, Select, message } from 'antd'
import { PlusOutlined, SendOutlined } from '@ant-design/icons'
import { warehouseApi } from '../../api/warehouse'

export default function ShippingPage() {
  const [boxes, setBoxes] = useState<any[]>([])
  const [pallets, setPallets] = useState<any[]>([])
  const [shipments, setShipments] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [shipModalVisible, setShipModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const [boxRes, palletRes, shipRes] = await Promise.all([
        warehouseApi.getBoxes(),
        warehouseApi.getPallets(),
        warehouseApi.getShipments()
      ])
      setBoxes(boxRes.data)
      setPallets(palletRes.data)
      setShipments(shipRes.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const shipmentColumns = [
    { title: '出货单号', dataIndex: 'shipment_number', key: 'shipment_number' },
    { title: '箱数', dataIndex: 'box_count', key: 'box_count' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => {
      const colors: Record<string, string> = { pending: 'orange', shipped: 'green', cancelled: 'red' }
      const texts: Record<string, string> = { pending: '待出货', shipped: '已出货', cancelled: '已取消' }
      return <Tag color={colors[s]}>{texts[s]}</Tag>
    }},
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at' },
    { title: '操作', key: 'action', render: (_: any, record: any) => (
      record.status === 'pending' && <Button type="primary" size="small" icon={<SendOutlined />}>发货</Button>
    )}
  ]

  const handleShipSubmit = async (values: any) => {
    try {
      message.success('出货单创建成功')
      setShipModalVisible(false)
      fetchData()
    } catch (err) {
      message.error('创建失败')
    }
  }

  return (
    <div>
      <Card 
        title="出货管理" 
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setShipModalVisible(true)}>创建出货单</Button>}
      >
        <Table 
          columns={shipmentColumns} 
          dataSource={shipments}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Card title="栈板状态" style={{ marginTop: 24 }}>
        <Table 
          columns={[
            { title: '栈板号', dataIndex: 'pallet_number', key: 'pallet_number' },
            { title: '箱数', dataIndex: 'box_count', key: 'box_count' },
            { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => {
              const colors: Record<string, string> = { loaded: 'blue', shipped: 'green', pending: 'orange' }
              const texts: Record<string, string> = { loaded: '已装载', shipped: '已发货', pending: '待装载' }
              return <Tag color={colors[s]}>{texts[s]}</Tag>
            }},
          ]}
          dataSource={pallets}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="创建出货单" open={shipModalVisible} onCancel={() => setShipModalVisible(false)} onOk={() => form.submit()}>
        <Form form={form} onFinish={handleShipSubmit} layout="vertical">
          <Form.Item name="shipment_number" label="出货单号" rules={[{ required: true }]}>
            <Input placeholder="例如: SH20240001" />
          </Form.Item>
          <Form.Item name="pallet_ids" label="栈板" rules={[{ required: true }]}>
            <Select mode="multiple" placeholder="选择栈板">
              {pallets.filter(p => p.status === 'loaded').map(p => <Select.Option key={p.id} value={p.id}>{p.pallet_number}</Select.Option>)}
            </Select>
          </Form.Item>
          <Form.Item name="destination" label="目的地">
            <Input placeholder="目的地" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
