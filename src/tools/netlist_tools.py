"""
Netlist tools for the circuit-analysis ReAct agent.

Only the active LangChain tools are kept here:
- element_transfer
- get_voltage
- get_current
"""

from typing import Any, Dict, List
import multiprocessing as mp
import os
import re
import signal
import warnings
from contextlib import contextmanager

from langchain_core.tools import tool

# Filter Lcapy normal warnings (remove voltage source for transfer function analysis)
warnings.filterwarnings("ignore", message="Removing voltage source")


# ============================================================
# Global Configuration
# ============================================================

try:
    LCAPY_TIMEOUT_SECONDS = int(os.getenv("LCAPY_TIMEOUT_SECONDS", "50"))
except Exception:
    LCAPY_TIMEOUT_SECONDS = 50


# ============================================================
# Timeout Error
# ============================================================

class TimeoutError(Exception):
    pass


@contextmanager
def _alarm_timeout(seconds: int, message: str):
    def _handle_timeout(signum, frame):
        raise TimeoutError(message)

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


# ============================================================
# LcapySolver - Core Circuit Analysis Engine
# ============================================================

class LcapySolver:
    """
    Lcapy-based circuit solver using native Lcapy methods.

    Reference: https://lcapy.readthedocs.io/en/latest/circuits.html

    Analysis domains:
    - s: Laplace (complex frequency) domain
    - t: time domain
    - f: Fourier (frequency) domain
    - omega/jomega: angular frequency domain
    """

    def __init__(self):
        self.circuit = None
        self.netlist = ""
        self.input_source = "V1"
        self.output_node = None

    def load_from_ir(self, ir_dict: Dict[str, Any]):
        """Load circuit from IR dictionary"""
        self.netlist = ir_dict.get("netlist", "")
        self.input_source = ir_dict.get("input_source")
        self.output_node = ir_dict.get("output_node")

        try:
            from lcapy import Circuit
            self.circuit = Circuit(self.netlist)

            # If no input_source specified, automatically detect independent source
            if not self.input_source:
                self.input_source = self._detect_input_source()

        except Exception as e:
            self.circuit = None
            raise ValueError(f"Failed to create circuit: {e}")

    def _detect_input_source(self) -> str:
        """Automatically detect independent source as input"""
        if not self.circuit:
            return "V1"

        # Find independent voltage source (V prefix)
        for name in self.circuit.elements:
            if name.startswith('V'):
                return name

        # Find independent current source (I prefix)
        for name in self.circuit.elements:
            if name.startswith('I'):
                return name

        return "V1"  # Default

    # --------------------------------------------------------
    # Basic Circuit Analysis: Voltage & Current
    # --------------------------------------------------------

    def get_voltage(self, target: str) -> Dict[str, Any]:
        """Get voltage at a node or across an element."""
        if not self.circuit:
            return {"success": False, "error": "Circuit not loaded"}

        try:
            # Use subprocess protection, prevent timeout
            code = f"str(cct['{target}'].V.s.simplify())"
            v_str = self._safe_call("get_voltage", code)
            return {
                "success": True,
                "target": target,
                "voltage": v_str,
            }
        except TimeoutError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_current(self, element: str) -> Dict[str, Any]:
        """Get current through an element."""
        if not self.circuit:
            return {"success": False, "error": "Circuit not loaded"}

        try:
            # Use subprocess protection, prevent timeout
            code = f"str(cct['{element}'].I.s.simplify())"
            i_str = self._safe_call("get_current", code)
            return {
                "success": True,
                "element": element,
                "current": i_str,
            }
        except TimeoutError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # --------------------------------------------------------
    # Transfer Function Analysis (call Lcapy directly, without multiprocessing)
    # --------------------------------------------------------

    def _compute_transfer(self, n1p, n1m, n2p, n2m):
        """Call Lcapy transfer() directly - use subprocess to handle recursion and timeout"""
        return self._safe_lcapy_call("transfer", n1p, n1m, n2p, n2m)

    def _compute_resistive_element_transfer(self, output_element: str, input_source: str) -> Dict[str, str]:
        """Compute V(output_element) / V(input_source) with symbolic MNA.

        Lcapy's transfer() can return misleading constants for some floating
        resistive subgraphs. This path solves the driven circuit directly.
        """
        import sympy as sp

        elements = []
        nodes = set()
        for raw_line in self.netlist.strip().splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if not line or line.startswith(("*", "#")):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            name = parts[0]
            kind = name[0].upper()
            if kind not in {"R", "V"}:
                raise ValueError("symbolic MNA fallback only supports R and V elements")
            n_plus, n_minus = str(parts[1]), str(parts[2])
            value = parts[3]
            elements.append((name, kind, n_plus, n_minus, value))
            nodes.update([n_plus, n_minus])

        output = next((elem for elem in elements if elem[0].lower() == output_element.lower()), None)
        source = next((elem for elem in elements if elem[0].lower() == input_source.lower()), None)
        if not output:
            raise ValueError(f"Output element not found: {output_element}")
        if not source or source[1] != "V":
            raise ValueError(f"Input voltage source not found: {input_source}")

        protected_nodes = {source[2], source[3], output[2], output[3], "0"}
        internal_graph = {}
        for name, kind, n_plus, n_minus, _ in elements:
            if kind != "R":
                continue
            for node, other in ((n_plus, n_minus), (n_minus, n_plus)):
                if node not in protected_nodes and other not in protected_nodes:
                    internal_graph.setdefault(node, set()).add(other)
                elif node not in protected_nodes:
                    internal_graph.setdefault(node, set())

        pruned_internal_nodes = set()
        seen_internal_nodes = set()
        for start in list(internal_graph):
            if start in seen_internal_nodes:
                continue
            stack = [start]
            component = set()
            boundaries = set()
            while stack:
                node = stack.pop()
                if node in seen_internal_nodes:
                    continue
                seen_internal_nodes.add(node)
                component.add(node)
                for _, kind, n_plus, n_minus, _ in elements:
                    if kind != "R" or node not in (n_plus, n_minus):
                        continue
                    other = n_minus if node == n_plus else n_plus
                    if other in protected_nodes:
                        boundaries.add(other)
                    elif other not in seen_internal_nodes:
                        stack.append(other)
            if len(boundaries) <= 1:
                pruned_internal_nodes.update(component)

        if pruned_internal_nodes:
            elements = [
                elem for elem in elements
                if elem[1] != "R" or (elem[2] not in pruned_internal_nodes and elem[3] not in pruned_internal_nodes)
            ]
            nodes = {node for _, _, n_plus, n_minus, _ in elements for node in (n_plus, n_minus)}

        if len(voltage_sources := [elem for elem in elements if elem[1] == "V"]) == 1:
            _, _, src_plus, src_minus, _ = source
            if src_plus == "0" or src_minus == "0":
                known_voltages = {"0": sp.Integer(0)}
                known_voltages[src_minus if src_plus == "0" else src_plus] = sp.Integer(-1 if src_plus == "0" else 1)
                unknown_nodes = sorted(
                    (node for node in nodes if node not in known_voltages),
                    key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item),
                )
                node_index = {node: idx for idx, node in enumerate(unknown_nodes)}
                if unknown_nodes:
                    A = sp.zeros(len(unknown_nodes), len(unknown_nodes))
                    b = sp.zeros(len(unknown_nodes), 1)
                    for _, kind, n_plus, n_minus, value in elements:
                        if kind != "R":
                            continue
                        conductance = 1 / sp.Symbol(value)
                        for node, other in ((n_plus, n_minus), (n_minus, n_plus)):
                            idx = node_index.get(node)
                            if idx is None:
                                continue
                            A[idx, idx] += conductance
                            other_idx = node_index.get(other)
                            if other_idx is not None:
                                A[idx, other_idx] -= conductance
                            else:
                                b[idx, 0] += conductance * known_voltages.get(other, 0)
                    solution = A.LUsolve(b)
                else:
                    solution = []

                def solved_voltage(node: str):
                    if node in known_voltages:
                        return known_voltages[node]
                    return solution[node_index[node], 0]

                _, _, out_plus, out_minus, _ = output
                tf = sp.factor(sp.cancel(solved_voltage(out_plus) - solved_voltage(out_minus)))
                return {
                    "transfer_function": str(tf),
                    "input_nodes": f"{src_plus}-{src_minus}",
                    "output_nodes": f"{out_plus}-{out_minus}",
                }

        node_names = sorted((node for node in nodes if node != "0"), key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item))
        node_index = {node: idx for idx, node in enumerate(node_names)}
        source_index = {elem[0]: len(node_names) + idx for idx, elem in enumerate(voltage_sources)}
        size = len(node_names) + len(voltage_sources)
        if size == 0:
            raise ValueError("No unknowns in circuit")

        A = sp.zeros(size, size)
        b = sp.zeros(size, 1)

        def node_var(node: str):
            return node_index.get(node)

        for name, kind, n_plus, n_minus, value in elements:
            p_idx = node_var(n_plus)
            m_idx = node_var(n_minus)
            if kind == "R":
                conductance = 1 / sp.Symbol(value)
                if p_idx is not None:
                    A[p_idx, p_idx] += conductance
                if m_idx is not None:
                    A[m_idx, m_idx] += conductance
                if p_idx is not None and m_idx is not None:
                    A[p_idx, m_idx] -= conductance
                    A[m_idx, p_idx] -= conductance
            elif kind == "V":
                v_idx = source_index[name]
                if p_idx is not None:
                    A[p_idx, v_idx] += 1
                    A[v_idx, p_idx] += 1
                if m_idx is not None:
                    A[m_idx, v_idx] -= 1
                    A[v_idx, m_idx] -= 1
                b[v_idx, 0] = 1 if name.lower() == input_source.lower() else 0

        solution = A.LUsolve(b)

        def voltage(node: str):
            idx = node_var(node)
            return 0 if idx is None else solution[idx, 0]

        _, _, src_plus, src_minus, _ = source
        _, _, out_plus, out_minus, _ = output
        tf = sp.factor(sp.cancel(voltage(out_plus) - voltage(out_minus)))
        return {
            "transfer_function": str(tf),
            "input_nodes": f"{src_plus}-{src_minus}",
            "output_nodes": f"{out_plus}-{out_minus}",
        }


    def _safe_call(self, operation_name: str, callable_code: str, timeout_sec=None) -> Any:
        """Generic safe Lcapy call, using subprocess to handle recursion limit and timeout.

        Args:
            operation_name: operation name (for error message)
            callable_code: Python code string to execute, with available variables:
                           - cct: Lcapy Circuit object
                           - The value of the last expression will be returned
            timeout_sec: timeout seconds

        Returns:
            Operation result (serialized to string or pickle-serializable object)
        """
        if timeout_sec is None:
            timeout_sec = LCAPY_TIMEOUT_SECONDS

        def _execute_in_process(netlist_str, code, result_queue):
            import sys
            sys.setrecursionlimit(100000)  # Increase recursion limit
            try:
                from lcapy import Circuit
                cct = Circuit(netlist_str)
                # Execute code, the value of the last expression will be eval returned
                result = eval(code, {"cct": cct, "__builtins__": __builtins__})
                # Try to serialize result
                result_queue.put(("success", result))
            except RecursionError:
                result_queue.put(("error", f"Circuit too complex: recursion depth exceeded during {operation_name}. Consider simplifying the circuit or using a different tool."))
            except Exception as e:
                result_queue.put(("error", str(e)))

        netlist_str = str(self.circuit)

        result_queue = mp.Queue()
        process = mp.Process(
            target=_execute_in_process,
            args=(netlist_str, callable_code, result_queue)
        )
        process.start()
        process.join(timeout=timeout_sec)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
            raise TimeoutError(
                f"{operation_name} TIMEOUT after {timeout_sec}s. "
                f"MUST SWITCH TOOL! Consider using a different analysis method."
            )

        if not result_queue.empty():
            status, result = result_queue.get()
            if status == "success":
                return result
            else:
                raise Exception(result)
        else:
            raise TimeoutError(f"{operation_name}: Process ended without result")

    def _safe_lcapy_call(self, method_name: str, n1p, n1m, n2p, n2m, timeout_sec=None):
        """Generic safe Lcapy call, using subprocess to handle recursion limit and timeout"""
        if timeout_sec is None:
            timeout_sec = LCAPY_TIMEOUT_SECONDS

        def _compute_in_process(netlist_str, method, n1p, n1m, n2p, n2m, result_queue):
            import sys
            sys.setrecursionlimit(100000)  # Increase recursion limit
            try:
                from lcapy import Circuit
                cct = Circuit(netlist_str)
                H = cct.transfer(n1p, n1m, n2p, n2m)
                result_queue.put(("success", str(H.simplify())))
            except RecursionError:
                result_queue.put(("error", "Circuit too complex: recursion depth exceeded. SWITCH TOOL! Use element_transfer, symbolic_solve, or calculate instead."))
            except Exception as e:
                result_queue.put(("error", str(e)))

        netlist_str = str(self.circuit)

        result_queue = mp.Queue()
        process = mp.Process(
            target=_compute_in_process,
            args=(netlist_str, method_name, n1p, n1m, n2p, n2m, result_queue)
        )
        process.start()
        process.join(timeout=timeout_sec)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
            raise TimeoutError(
                f"{method_name} TIMEOUT after {timeout_sec}s. "
                f"MUST SWITCH TOOL! Use: element_transfer, symbolic_solve, calculate, or loop_gain instead. "
                f"DO NOT retry same tool!"
            )

        if not result_queue.empty():
            status, result = result_queue.get()
            if status == "success":
                return result  # Return string
            else:
                raise Exception(result)
        else:
            raise TimeoutError("Process ended without result")

    def element_transfer_function(self, output_element: str,
                                   input_source: str = None) -> Dict[str, Any]:
        """Compute transfer function from input source to an element voltage.

        H(s) = V_element / V_input

        This is useful when the question asks for transfer function TO an element
        (e.g., "transfer function from V1 to R3").

        Args:
            output_element: Element name (e.g., "R1", "R3", "C1")
            input_source: Input source name (default: self.input_source)
        """
        if not self.circuit:
            return {"success": False, "error": "Circuit not loaded"}

        input_source = input_source or self.input_source

        try:
            with _alarm_timeout(5, "symbolic MNA fallback timed out"):
                mna_result = self._compute_resistive_element_transfer(output_element, input_source)
            return {
                "success": True,
                "transfer_function": mna_result["transfer_function"],
                "input": input_source,
                "input_nodes": mna_result["input_nodes"],
                "output_element": output_element,
                "output_nodes": mna_result["output_nodes"],
                "method": "symbolic_mna",
            }
        except Exception:
            pass

        
        # Get input source nodes
        src_elem = self.circuit[input_source]
        src_nodes = list(src_elem.nodes)
        n1p, n1m = str(src_nodes[0]), str(src_nodes[1])

        # Get output element nodes
        out_elem = self.circuit[output_element]
        out_nodes = list(out_elem.nodes)
        n2p, n2m = str(out_nodes[0]), str(out_nodes[1])

        # Compute transfer function using element nodes
        # Try integer conversion for transfer()
        n1p_v = int(n1p) if n1p.isdigit() else n1p
        n1m_v = int(n1m) if n1m.isdigit() else n1m
        n2p_v = int(n2p) if n2p.isdigit() else n2p
        n2m_v = int(n2m) if n2m.isdigit() else n2m

        tf = self._compute_transfer(n1p_v, n1m_v, n2p_v, n2m_v)

        return {
            "success": True,
            "transfer_function": tf,
            "input": input_source,
            "input_nodes": f"{n1p}-{n1m}",
            "output_element": output_element,
            "output_nodes": f"{n2p}-{n2m}",
            "method": "transfer",
        }
         
           

    def _create_passive_circuit(self):
        """Create a circuit copy with independent sources removed.

        For two-port parameter calculation, independent sources should be:
        - Voltage sources: short-circuited (removed, nodes connected)
        - Current sources: open-circuited (removed)

        Returns the passive circuit or None if creation fails.
        """
        try:
            from lcapy import Circuit

            passive_lines = []
            removed_sources = []

            for name, elem in self.circuit.elements.items():
                # Skip independent voltage sources (V prefix) and current sources (I prefix)
                # But keep dependent sources (E, F, G, H prefixes)
                if name.startswith('V') or name.startswith('I'):
                    # Independent sources have values like "V1 1 0 1" or "V1 1 0 {Vs}"
                    removed_sources.append(name)

                    # For voltage sources, we could add a wire (short circuit)
                    # But lcapy handles missing sources gracefully, so we just skip
                    continue
                else:
                    # Keep all other elements
                    passive_lines.append(str(elem))

            if not passive_lines:
                return None, []

            passive_netlist = '\n'.join(passive_lines)
            passive_circuit = Circuit(passive_netlist)

            return passive_circuit, removed_sources

        except Exception as e:
            return None, []

    def _has_symbolic_params(self) -> bool:
        """Check if netlist component value contains symbolic parameters

        Note: Check component value, not component name!
        For example: Ro1 7 4 ro1 → ro1 is symbolic parameter (needs replacement)
                     Ro1 7 4 3200 → 3200 is numeric (no replacement)
        """
        import re
        # Symbolic parameter pattern
        symbolic_patterns = [
            r'\bgm\d*\b',      # gm, gm1, gm2...
            r'\bro\d*\b',      # ro, ro1, ro2...
            r'\brpi\d*\b',     # rpi, rpi1...
            r'\brds\d*\b',     # rds, rds1... (drain-source resistance)
            r'\bgds\d*\b',     # gds, gds1...
            r'\b[Cc]gs\d*\b',  # Cgs, cgs1...
            r'\b[Cc]gd\d*\b',  # Cgd, cgd1...
            r'\bbeta\d*\b',    # beta, beta1...
            r'\brd\d*\b',      # rd, rd1...
        ]

        # Check component value (last field)
        for line in self.netlist.strip().split('\n'):
            line = line.split(';')[0].strip()  # Remove comments
            if not line or line.startswith('*'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            # Get component value (usually the last or second last field)
            # Format: Rname Np Nm value or Gname Np Nm Ncp Ncm value
            component_name = parts[0].upper()

            # Determine value position based on component type
            if component_name[0] in ['R', 'C', 'L', 'G'] and len(parts) >= 4:
                # R/C/L: Rname Np Nm value
                # G (2-node conductance): Gname Np Nm value
                # G (4-node VCCS): Gname Np Nm Ncp Ncm value
                if len(parts) == 4:
                    value_field = parts[3]
                elif len(parts) >= 6:
                    value_field = parts[5]
                else:
                    continue

                # Check if value is symbolic parameter
                for pattern in symbolic_patterns:
                    if re.match(pattern, value_field, re.IGNORECASE):
                        return True

        return False

    def _get_remaining_symbolic_params(self) -> List[str]:
        """Get list of remaining symbolic parameters in netlist component value

        Note: Only check component value, not component name!
        """
        import re

        symbolic_patterns = [
            r'\bgm\d*\b',      # gm, gm1, gm2...
            r'\bro\d*\b',      # ro, ro1, ro2...
            r'\brpi\d*\b',     # rpi, rpi1...
            r'\brds\d*\b',     # rds, rds1...
            r'\bgds\d*\b',     # gds, gds1...
            r'\b[Cc]gs\d*\b',  # Cgs, cgs1...
            r'\b[Cc]gd\d*\b',  # Cgd, cgd1...
            r'\bbeta\d*\b',    # beta, beta1...
            r'\brd\d*\b',      # rd, rd1...
        ]

        found_params = set()

        # Check component value line by line
        for line in self.netlist.strip().split('\n'):
            line = line.split(';')[0].strip()  # Remove comments
            if not line or line.startswith('*'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            component_name = parts[0].upper()

            # Determine value position based on component type
            value_fields = []
            if component_name[0] in ['R', 'C', 'L']:
                # R/C/L: Rname Np Nm value
                if len(parts) >= 4:
                    value_fields = [parts[3]]
            elif component_name[0] == 'G':
                # G (2-node): Gname Np Nm value
                # G (4-node VCCS): Gname Np Nm Ncp Ncm value
                if len(parts) == 4:
                    value_fields = [parts[3]]
                elif len(parts) >= 6:
                    value_fields = [parts[5]]
            elif component_name[0] == 'E':
                # E (VCVS): Ename Np Nm Ncp Ncm gain [Rout]
                if len(parts) >= 6:
                    value_fields = parts[5:]  # gain and optional Rout

            # Check if value field matches symbolic parameter
            for value_field in value_fields:
                for pattern in symbolic_patterns:
                    if re.match(pattern, value_field, re.IGNORECASE):
                        found_params.add(value_field)

        return sorted(found_params, key=lambda x: (x.lower(), x))


# ============================================================
# LangChain Tools Factory
# ============================================================

def create_netlist_tools(ir_dict: Dict[str, Any]):
    """Create LangChain tools for netlist/circuit analysis.

    Returns a curated set of tools based on Lcapy capabilities.
    """

    solver = LcapySolver()
    try:
        solver.load_from_ir(ir_dict)
    except Exception as e:
        @tool
        def error_tool() -> str:
            """Report circuit loading error"""
            return f"Failed to load circuit: {e}"
        return [error_tool]


    # ========== Basic Analysis ==========

    @tool
    def get_voltage(target: str) -> str:
        """Get node voltage or element voltage.

        Use cases:
        - "What is the voltage of V2?" → get_voltage("2")
        - "What is the voltage across R1?" → get_voltage("R1")
        - "What is the voltage of node 3?" → get_voltage("3")

        Args:
            target: Node name (e.g., "2", "3") or element name (e.g., "R1", "C1")
        """
        result = solver.get_voltage(target)
        if result["success"]:
            voltage_str = str(result['voltage'])
            # Check if the result is empty (no source circuit)
            if 'SuperpositionVoltage({})' in voltage_str or voltage_str == '{}' or voltage_str == '0':
                return (f"V({target}) = 0 (No sources in circuit!)\n"
                        f"⚠️ Circuit has no source!\n"
                        f"Check: 0V sensing source (Vsense) cannot be used as INPUT_SOURCE.\n"
                        f"Need to add a real input source (e.g., Vin 1 0 {{vin}}) to excite the circuit.")
            return f"V({target}) = {voltage_str}"
        return f"Error: {result['error']}"

    @tool
    def get_current(element: str) -> str:
        """Get current through [two-terminal element].

        ⚠️ Only supports two-terminal elements: R, L, C, V, I (resistor/inductor/capacitor/voltage source/current source)
        ❌ Not supported: Controlled sources (E, F, G, H) - these are four-terminal or special elements

        Use cases:
        - "What is the current through R1?" → get_current("R1")
        - "I_C1 = ?" → get_current("C1")
        - "What is the current of voltage source V1?" → get_current("V1")

        Args:
            element: Two-terminal element name (e.g., "R1", "C1", "L1", "V1")
        """
        result = solver.get_current(element)
        if result["success"]:
            return f"I({element}) = {result['current']}"
        return f"Error: Cannot get current for '{element}'. Only 2-terminal elements (R/L/C/V/I) are supported. Controlled sources (E/F/G/H) are not supported."


    @tool
    def element_transfer(output_element: str, input_source: str = "") -> str:
        """Calculate H(s) = V_element / V_in (element voltage)

        ⚠️ Only use when the question explicitly asks for the voltage across an element!

        Examples (must mention the element name):
        - "voltage across R3" → element_transfer("R3","V1")
        - "V_R1/V_in" → element_transfer("R1","V1")
        - "Voltage across R2" → element_transfer("R2","V1")

        ❌ Do not use for general transfer function problems, use node_transfer instead

        Args:
            output_element: Element name (e.g., "R1", "C1", "L1")
            input_source: Input source
        """
        src = input_source if input_source else None
        result = solver.element_transfer_function(output_element=output_element, input_source=src)
        if result["success"]:
            return f"H(s) = {result['transfer_function']}"
        return f"Error: {result['error']}"

    @tool
    def solve_node_voltage(node: str) -> str:
        """Solve for node voltage Vn<node>(s) in the s-domain."""
        target = str(node).strip()
        if not target:
            target = "1"
        target = target.removeprefix("Vn").removeprefix("vn")
        result = solver.get_voltage(target)
        if result["success"]:
            return f"Vn{target}(s) = {result['voltage']}"
        return f"Error: {result['error']}"

    @tool
    def solve_branch_current(target: str) -> str:
        """Solve for branch current such as iv1, il1, or ieint1 in the s-domain."""
        raw = str(target).strip() or "iv1"
        match = re.fullmatch(r"[Vv](\d+)", raw)
        element = f"V{match.group(1)}" if match else raw
        if raw.lower().startswith("iv"):
            element = f"V{raw[2:]}"
        elif raw.lower().startswith("il"):
            element = f"L{raw[2:]}"
        result = solver.get_current(element)
        if result["success"]:
            return f"{raw}(s) = {result['current']}"
        return f"Error: {result['error']}"


    return  [

        # ============================================================
        # Transfer function analysis
        # ============================================================
        element_transfer,       # H(s) = V_element/V_in

        # # ============================================================
        # # Basic tools
        # # ============================================================
        get_voltage,            # Node/element voltage V(x)
        get_current,            # Element current I(x)
        solve_node_voltage,
        solve_branch_current,

    ]
