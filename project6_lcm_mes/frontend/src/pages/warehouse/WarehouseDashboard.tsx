import { useState, useEffect } from 'react'
import { Row, Col, Card, Statistic, Table, Tag, Button, Modal, Form, Input, InputNumber, Select, message } from 'antd'
import { PlusOutlined, InboxOutlined, SendOutlined, BoxPlotOutlined } from '@ant-design/icons'
import { warehouseApi } from '../../api/warehouse'
import { materialApi } from '../../api/material'

export default function WarehouseDashboard() {
  const [boxes, setBoxes] = useState<any[]>([])
  const [pallets, setPallets] = useState<any[]>([])
  const [materials, setMaterials] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [lotModalVisible, setLotModalVisible] = useState(false)
  const [boxModalVisible, setBoxModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const [boxRes, palletRes, matRes] = await Promise.all([
        warehouseApi.getBoxes(),
        warehouseApi.getPallets(),
        materialApi.getMaterials()
      ])
      setBoxes(boxRes.data)
      setPallets(palletRes.data)
      setMaterials(matRes.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const boxColumns = [
    { title: '箱号', dataIndex: 'box_number', key: 'box_number' },
    { title: '数量', dataIndex: 'quantity', key: 'quantity' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => {
      const colors: Record<string, string> = { packed: 'blue', shipped: 'green', pending: 'orange' }
      const texts: Record<string, string> = { packed: '已装箱', shipped: '已发货', pending: '待包装' }
      return <Tag color={colors[s]}>{texts[s]}</Tag>
    }},
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at' },
  ]

  const palletColumns = [
    { title: '栈板号', dataIndex: 'pallet_number', key: 'pallet_number' },
    { title: '箱数', dataIndex: 'box_count', key: 'box_count' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => {
      const colors: Record<string, string> = { loaded: 'blue', shipped: 'green', pending: 'orange' }
      const texts: Record<string, string> = { loaded: '已装载', shipped: '已发货', pending: '待装载' }
      return <Tag color={colors[s]}>{texts[s]}</Tag>
    }},
  ]

  const handleLotSubmit = async (values: any) => {
    try {
      message.success('Lot录入成功')
      setLotModalVisible(false)
      fetchData()
    } catch (err) {
      message.error('录入失败')
    }
  }

  const handleBoxSubmit = async (values: any) => {
    try {
      message.success('装箱成功')
      setBoxModalVisible(false)
      fetchData()
    } catch (err) {
      message.error('装箱失败')
    }
  }

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="待入库Lot"
              value={5}
              prefix={<InboxOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已装箱"
              value={156}
              prefix={<BoxPlotOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已出货栈板"
              value={12}
              prefix={<SendOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="库存周转天数"
              value={7}
              suffix="天"
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card 
            title="装箱管理" 
            extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setBoxModalVisible(true)}>装箱</Button>}
          >
            <Table 
              columns={boxColumns} 
              dataSource={boxes}
              loading={loading}
              rowKey="id"
              pagination={{ pageSize: 5 }}
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card 
            title="栈板管理" 
            extra={<Button type="primary" icon={<PlusOutlined />}>出货</Button>}
          >
            <Table 
              columns={palletColumns} 
              dataSource={pallets}
              loading={loading}
              rowKey="id"
              pagination={{ pageSize: 5 }}
            />
          </Card>
        </Col>
      </Row>

      <Card 
        title="入库管理" 
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setLotModalVisible(true)}>Lot入库</Button>}
      >
        <Table 
          columns={[
            { title: '物料', dataIndex: 'name', key: 'name' },
            { title: '编码', dataIndex: 'code', key: 'code' },
            { title: '库存', dataIndex: 'quantity', key: 'quantity' },
            { title: '单位', dataIndex: 'unit', key: 'unit' },
          ]}
          dataSource={materials}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="Lot入库" open={lotModalVisible} onCancel={() => setLotModalVisible(false)} onOk={() => form.submit()}>
        <Form form={form} onFinish={handleLotSubmit} layout="vertical">
          <Form.Item name="lot_number" label="Lot号" rules={[{ required: true }]}>
            <Input placeholder="例如: LOT20240001" />
          </Form.Item>
          <Form.Item name="material_id" label="物料" rules={[{ required: true }]}>
            <Select placeholder="选择物料">
              {materials.map(m => <Select.Option key={m.id} value={m.id}>{m.name}</Select.Option>)}
            </Select>
          </Form.Item>
          <Form.Item name="quantity" label="数量" rules={[{ required: true }]}>
            <InputNumber style={{ width: '100%' }} min={1} />
          </Form.Item>
        </Form>
      </Modal>

      <Modal title="装箱" open={boxModalVisible} onCancel={() => setBoxModalVisible(false)} onOk={() => form.submit()}>
        <Form form={form} onFinish={handleBoxSubmit} layout="vertical">
          <Form.Item name="box_number" label="箱号" rules={[{ required: true }]}>
            <Input placeholder="例如: BOX20240001" />
          </Form.Item>
          <Form.Item name="product_codes" label="产品编码列表" rules={[{ required: true }]}>
            <Input.TextArea placeholder="输入产品编码，每行一个" rows={6} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
