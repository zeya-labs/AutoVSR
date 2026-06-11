(function () {
  const NS = 'http://www.w3.org/2000/svg';

  function svgEl(tag, attrs = {}, text = '') {
    const el = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
    if (text) el.textContent = text;
    return el;
  }

  function endpoint(from, to) {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const halfW = Math.max(from.w || 60, 20) / 2;
    const halfH = Math.max(from.h || 32, 20) / 2;
    if (dx === 0 && dy === 0) return { x: from.x, y: from.y };
    const tx = Math.abs(dx) > 0 ? halfW / Math.abs(dx) : Infinity;
    const ty = Math.abs(dy) > 0 ? halfH / Math.abs(dy) : Infinity;
    const t = Math.min(tx, ty, 1);
    return { x: from.x + dx * t, y: from.y + dy * t };
  }

  function shortId(prefix) {
    return `${prefix}_${Math.random().toString(36).slice(2, 8)}`;
  }

  function renderEditor(root, layout) {
    layout.nodes = layout.nodes || [];
    layout.edges = layout.edges || [];
    layout.groups = layout.groups || [];
    const nodes = new Map(layout.nodes.map((node) => [node.id, { ...node }]));
    const edges = layout.edges.map((edge) => ({ ...edge }));
    const width = Math.max(Number(layout.width || 900), 600);
    const height = Math.max(Number(layout.height || 600), 360);
    let selected = null;
    let connectSource = null;

    const svg = svgEl('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img' });
    const defs = svgEl('defs');
    const marker = svgEl('marker', {
      id: `arrow-${root.dataset.case}`, viewBox: '0 0 10 10', refX: '9', refY: '5',
      markerWidth: '7', markerHeight: '7', orient: 'auto-start-reverse'
    });
    marker.appendChild(svgEl('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#1f5fae' }));
    defs.appendChild(marker);
    svg.appendChild(defs);
    const groupLayer = svgEl('g', { class: 'groups' });
    const edgeLayer = svgEl('g', { class: 'edges' });
    const nodeLayer = svgEl('g', { class: 'nodes' });
    svg.append(groupLayer, edgeLayer, nodeLayer);
    root.querySelector('.editable-graph').replaceChildren(svg);

    const status = root.querySelector('.save-status');
    const labelInput = root.querySelector('.edit-label');
    const kindSelect = root.querySelector('.edit-kind');
    const autoTidyButton = root.querySelector('.auto-tidy');
    const undoTidyButton = root.querySelector('.undo-tidy');
    const addNodeButton = root.querySelector('.add-node');
    const connectButton = root.querySelector('.connect-edge');
    const deleteButton = root.querySelector('.delete-selected');
    const saveButton = root.querySelector('.save-graph');

    function setStatus(text) {
      status.textContent = text || '';
    }

    let tidyBackup = null;

    function backupNodePositions() {
      tidyBackup = Array.from(nodes.values()).map((node) => ({ id: node.id, x: node.x, y: node.y }));
    }

    function autoTidy() {
      backupNodePositions();
      const ids = Array.from(nodes.keys());
      const incoming = new Map(ids.map((id) => [id, 0]));
      const outgoing = new Map(ids.map((id) => [id, []]));
      for (const edge of edges) {
        if (!nodes.has(edge.source) || !nodes.has(edge.target)) continue;
        if (edge.kind === 'feedback' || edge.kind === 'annotation_link') continue;
        outgoing.get(edge.source).push(edge.target);
        incoming.set(edge.target, (incoming.get(edge.target) || 0) + 1);
      }
      const queue = ids.filter((id) => (incoming.get(id) || 0) === 0);
      const rank = new Map(ids.map((id) => [id, 0]));
      while (queue.length) {
        const id = queue.shift();
        const base = rank.get(id) || 0;
        for (const next of outgoing.get(id) || []) {
          rank.set(next, Math.max(rank.get(next) || 0, base + 1));
          incoming.set(next, (incoming.get(next) || 0) - 1);
          if ((incoming.get(next) || 0) === 0) queue.push(next);
        }
      }
      for (const edge of edges) {
        if (!nodes.has(edge.source) || !nodes.has(edge.target)) continue;
        if (edge.kind !== 'feedback' && edge.kind !== 'annotation_link') {
          rank.set(edge.target, Math.max(rank.get(edge.target) || 0, (rank.get(edge.source) || 0) + 1));
        }
      }
      const ranks = new Map();
      for (const id of ids) {
        const value = Math.min(rank.get(id) || 0, 8);
        if (!ranks.has(value)) ranks.set(value, []);
        ranks.get(value).push(nodes.get(id));
      }
      const marginX = 70;
      const marginY = 44;
      const sortedRanks = Array.from(ranks.keys()).sort((a, b) => a - b);
      const usableW = Math.max(width - marginX * 2, 400);
      const stepX = sortedRanks.length > 1 ? usableW / (sortedRanks.length - 1) : usableW / 2;
      for (const [rankIndex, rankValue] of sortedRanks.entries()) {
        const items = ranks.get(rankValue).sort((a, b) => a.y - b.y);
        const minGap = 58;
        let lastY = marginY - minGap;
        for (const node of items) {
          node.x = marginX + rankIndex * stepX;
          node.y = Math.max(marginY, Math.min(height - marginY, node.y));
          if (node.y - lastY < minGap) node.y = lastY + minGap;
          lastY = node.y;
        }
        const overflow = lastY - (height - marginY);
        if (overflow > 0) {
          for (const node of items) node.y -= overflow;
        }
      }
      draw();
      setStatus('auto tidy applied with current vertical order preserved');
    }

    function select(type, id) {
      selected = type && id ? { type, id } : null;
      connectSource = null;
      connectButton.classList.remove('active');
      const item = selected?.type === 'node' ? nodes.get(selected.id) : edges.find((edge) => edge.id === selected?.id);
      labelInput.value = item?.label || '';
      kindSelect.value = item?.kind || 'signal';
      draw();
    }

    function selectedNode() {
      return selected?.type === 'node' ? nodes.get(selected.id) : null;
    }

    function selectedEdge() {
      return selected?.type === 'edge' ? edges.find((edge) => edge.id === selected.id) : null;
    }

    function draw() {
      groupLayer.replaceChildren();
      edgeLayer.replaceChildren();
      nodeLayer.replaceChildren();
      for (const group of layout.groups || []) {
        const groupNodes = (group.nodes || []).map((id) => nodes.get(id)).filter(Boolean);
        if (!groupNodes.length) continue;
        const pad = 24;
        const minX = Math.min(...groupNodes.map((node) => node.x - node.w / 2)) - pad;
        const minY = Math.min(...groupNodes.map((node) => node.y - node.h / 2)) - pad;
        const maxX = Math.max(...groupNodes.map((node) => node.x + node.w / 2)) + pad;
        const maxY = Math.max(...groupNodes.map((node) => node.y + node.h / 2)) + pad;
        const g = svgEl('g', { class: 'editable-group', 'data-id': group.id || '' });
        g.append(
          svgEl('rect', {
            x: minX, y: minY, width: maxX - minX, height: maxY - minY,
            rx: 8, fill: '#ffffff', stroke: '#9aa7b8', 'stroke-width': 1.4,
            'stroke-dasharray': '7 4'
          }),
          svgEl('text', {
            x: (minX + maxX) / 2, y: minY + 16,
            'font-size': 12, 'font-family': 'Helvetica, Arial, sans-serif',
            'text-anchor': 'middle', fill: '#475569'
          }, group.label || group.id || 'group')
        );
        groupLayer.appendChild(g);
      }
      for (const edge of edges) {
        const source = nodes.get(edge.source);
        const target = nodes.get(edge.target);
        if (!source || !target) continue;
        const a = endpoint(source, target);
        const b = endpoint(target, source);
        const g = svgEl('g', { class: `editable-edge ${selected?.type === 'edge' && selected.id === edge.id ? 'selected' : ''}`, 'data-id': edge.id });
        const line = svgEl('line', {
          x1: a.x, y1: a.y, x2: b.x, y2: b.y,
          stroke: edge.color || '#1f5fae', 'stroke-width': selected?.id === edge.id ? 3 : 1.8,
          'marker-end': edge.direction === 'undirected' ? '' : `url(#arrow-${root.dataset.case})`
        });
        if (edge.kind === 'feedback' || edge.kind === 'annotation_link') line.setAttribute('stroke-dasharray', '6 4');
        const label = svgEl('text', {
          x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 - 5,
          'font-size': 11, 'font-family': 'Helvetica, Arial, sans-serif',
          fill: edge.color || '#1f5fae', 'text-anchor': 'middle'
        }, edge.label || edge.kind || '');
        g.append(line, label);
        g.addEventListener('pointerdown', (event) => {
          event.stopPropagation();
          select('edge', edge.id);
        });
        edgeLayer.appendChild(g);
      }

      for (const node of nodes.values()) {
        const isSelected = selected?.type === 'node' && selected.id === node.id;
        const g = svgEl('g', {
          class: `editable-node ${isSelected ? 'selected' : ''}`,
          'data-id': node.id,
          transform: `translate(${node.x},${node.y})`
        });
        const shapeAttrs = {
          fill: node.kind === 'operator' ? '#fff7ed' : '#f8fafc',
          stroke: isSelected ? '#b42318' : '#334155',
          'stroke-width': isSelected ? 2.4 : 1.4
        };
        const shape = node.shape === 'circle'
          ? svgEl('ellipse', { ...shapeAttrs, cx: 0, cy: 0, rx: Math.max(node.w / 2, 16), ry: Math.max(node.h / 2, 16) })
          : svgEl('rect', { ...shapeAttrs, x: -node.w / 2, y: -node.h / 2, width: node.w, height: node.h, rx: 5 });
        const text = svgEl('text', {
          x: 0, y: 4, 'font-size': 11, 'font-family': 'Helvetica, Arial, sans-serif',
          'text-anchor': 'middle', fill: '#18212f'
        }, node.label || node.id);
        g.append(svgEl('title', {}, node.id), shape, text);
        nodeLayer.appendChild(g);
      }
    }

    function pointerPos(event) {
      const pt = svg.createSVGPoint();
      pt.x = event.clientX;
      pt.y = event.clientY;
      return pt.matrixTransform(svg.getScreenCTM().inverse());
    }

    let drag = null;
    nodeLayer.addEventListener('pointerdown', (event) => {
      const g = event.target.closest('.editable-node');
      if (!g) return;
      const node = nodes.get(g.dataset.id);
      if (!node) return;
      event.stopPropagation();
      if (connectSource) {
        if (connectSource !== node.id) {
          edges.push({
            id: shortId('e'),
            source: connectSource,
            target: node.id,
            label: '',
            kind: 'signal',
            direction: 'directed',
            color: '#1f5fae'
          });
          setStatus(`connected ${connectSource} -> ${node.id}`);
        }
        select(null, null);
        return;
      }
      select('node', node.id);
      const pos = pointerPos(event);
      drag = { node, dx: pos.x - node.x, dy: pos.y - node.y };
      g.setPointerCapture(event.pointerId);
    });

    nodeLayer.addEventListener('pointermove', (event) => {
      if (!drag) return;
      const pos = pointerPos(event);
      drag.node.x = pos.x - drag.dx;
      drag.node.y = pos.y - drag.dy;
      draw();
    });

    nodeLayer.addEventListener('pointerup', () => {
      drag = null;
    });

    svg.addEventListener('pointerdown', () => {
      select(null, null);
    });

    addNodeButton.addEventListener('click', () => {
      const id = shortId('n');
      nodes.set(id, {
        id,
        label: 'new node',
        kind: 'functional_block',
        shape: 'box',
        x: width / 2,
        y: height / 2,
        w: 90,
        h: 36
      });
      select('node', id);
      setStatus(`added ${id}`);
    });

    autoTidyButton.addEventListener('click', autoTidy);
    undoTidyButton.addEventListener('click', () => {
      if (!tidyBackup) {
        setStatus('nothing to undo');
        return;
      }
      for (const item of tidyBackup) {
        const node = nodes.get(item.id);
        if (node) {
          node.x = item.x;
          node.y = item.y;
        }
      }
      tidyBackup = null;
      draw();
      setStatus('tidy undone');
    });

    connectButton.addEventListener('click', () => {
      const node = selectedNode();
      if (!node) {
        setStatus('select a source node first');
        return;
      }
      connectSource = node.id;
      connectButton.classList.add('active');
      setStatus(`select target node for ${node.id}`);
    });

    deleteButton.addEventListener('click', () => {
      if (!selected) return;
      if (selected.type === 'node') {
        nodes.delete(selected.id);
        for (let i = edges.length - 1; i >= 0; i--) {
          if (edges[i].source === selected.id || edges[i].target === selected.id) edges.splice(i, 1);
        }
      } else if (selected.type === 'edge') {
        const index = edges.findIndex((edge) => edge.id === selected.id);
        if (index >= 0) edges.splice(index, 1);
      }
      setStatus(`deleted ${selected.id}`);
      select(null, null);
    });

    labelInput.addEventListener('change', () => {
      const item = selectedNode() || selectedEdge();
      if (!item) return;
      item.label = labelInput.value;
      draw();
    });

    kindSelect.addEventListener('change', () => {
      const item = selectedNode() || selectedEdge();
      if (!item) return;
      item.kind = kindSelect.value;
      if (selected?.type === 'node') item.shape = kindSelect.value === 'operator' ? 'circle' : 'box';
      if (selected?.type === 'edge') {
        item.color = kindSelect.value === 'feedback' ? '#b42318' : kindSelect.value === 'bus' ? '#7a4cc2' : '#1f5fae';
      }
      draw();
    });

    function snapshotLayout() {
      return {
        ...layout,
        nodes: Array.from(nodes.values()).map((node) => ({ ...node })),
        edges: edges.map((edge) => ({ ...edge }))
      };
    }

    function serializeSvg() {
      const clone = svg.cloneNode(true);
      clone.setAttribute('xmlns', NS);
      return new XMLSerializer().serializeToString(clone);
    }

    saveButton.addEventListener('click', async () => {
      setStatus('saving...');
      try {
        const response = await fetch('/api/save_graph', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            case_id: root.dataset.case,
            layout_path: root.dataset.layout,
            layout: snapshotLayout(),
            svg: serializeSvg()
          })
        });
        const responseText = await response.text();
        let result = {};
        try {
          result = responseText ? JSON.parse(responseText) : {};
        } catch (parseError) {
          throw new Error(`server returned non-JSON response (${response.status}): ${responseText.slice(0, 120)}`);
        }
        if (!responseText) {
          throw new Error(`server returned empty response (${response.status}). Open the report with scripts/serve_eda_block_report.py, not a plain static/file server.`);
        }
        if (!response.ok || !result.ok) throw new Error(result.error || response.statusText);
        setStatus(`saved: ${result.layout_path}, ${result.svg_path}`);
      } catch (error) {
        setStatus(`save failed: ${error.message}`);
      }
    });

    draw();
  }

  for (const root of document.querySelectorAll('.graph-editor[data-layout]')) {
    fetch(root.dataset.layout)
      .then((response) => response.json())
      .then((layout) => renderEditor(root, layout))
      .catch((error) => {
        root.querySelector('.editable-graph').textContent = `failed to load editable graph: ${error}`;
      });
  }
})();
