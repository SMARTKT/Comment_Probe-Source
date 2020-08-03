import clang.cindex as cl
from clang.cindex import Index
from pprint import pprint
from optparse import OptionParser, OptionGroup
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree as ET
from xml.dom import minidom

cl.Config.set_library_file("/usr/lib64/llvm/libclang.so")

# func_node = None
# def get_info(node):
#   children = [get_info(c) for c in node.get_children()]
#   return {  'usr' : node.get_usr(),
#             'spelling' : node.spelling,
#             'location' : node.location,
#             'extent.start' : node.extent.start,
#             'extent.end' : node.extent.end,
#             'is_definition' : node.is_definition(),
#             'children' : children }
#
# def get_info(node, parent):
#     try:
#         sub = SubElement(parent, str(node.kind).split('.')[-1])
#     except:
#         sub = SubElement(parent, "Unknown")
#     # print(dir(node.extent.start))
#     for attr in dir(node):
#         try:
#             sub.set(str(attr), str(getattr(node, attr)))
#         except:
#             sub.set(str(attr), "None")
#     try:
#         for c in node.get_children():
#             try:
#                 children.append(get_info(c, sub))
#             except:
#                 continue
#     except:
#         return parent
#     # children = [get_info(c, sub) for c in node.get_children()]
#     return parent

def get_info(node, parent):
  global func_node
  try:
    sub = SubElement(parent, str(node.kind.name))
  except:
    sub = SubElement(parent, "Unknown")
  sub.set('id', str(node.hash))
  sub.set('parent_id', str(0) if node.semantic_parent is None else str(node.semantic_parent.hash))
  sub.set('usr', "None" if node.get_usr() is None else str(node.get_usr()))
  sub.set('spelling', "None" if node.spelling is None else str(node.spelling))
  sub.set('location', "None" if (node.location is None or node.location.file is None) else str(node.location.file)+"["+str(node.location.line)+"]")
  sub.set('linenum', "None" if (node.location.line is None) else str(node.location.line))
  sub.set('extent.start', "None" if (node.extent.start is None or node.extent.start.file is None) else str(node.extent.start.file)+"["+ str(node.extent.start.line) + "]")
  sub.set('extent.end', "None" if (node.extent.end is None or node.extent.end.file is None) else str(node.extent.end.file)+"["+ str(node.extent.end.line) + "]")
  sub.set('is_definition', str(node.is_definition()))
  sub.set('type', str(node.type.spelling))
  children = [get_info(c, sub) for c in node.get_children()]
  return parent

def do_something_with_try_statements(node):
    global root
    # prints the node
    print("Printing all THROW EXPRessions and the name, location and children")
    for c in root.findall(".//CXX_THROW_EXPR"):
        print(c.attrib['extent.start'])
        # To access definations inside a THROW EXPRESSIONS, use XML parser techniques

    print("\n")

    print("Printing all TRY and the name, location and children")
    for c in root.findall(".//CXX_TRY_STMT"):
        print(c.attrib['extent.start'])
        # To access definations inside a TRY EXPRESSIONS, use XML parser techniques
        # a TRY STMT has CATCH STMT as its children


parser = OptionParser("usage: %prog [options] {filename} [clang-args*]")
parser.disable_interspersed_args()
(opts, args) = parser.parse_args()

index = Index.create()
tu = index.parse(None, args)

if not tu:
  parser.error("unable to load input")

root = Element("STATICROOT")
root = get_info(tu.cursor, root)
root.set('id', str(0))
# print(ET.tostring(root, encoding='utf8').decode('utf8'))
xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
with open(str(args[0].split('.')[0])+"_clang.xml", "w") as f:
    f.write(xmlstr)
print("The entire XML parse has been written at ", str(args[0].split('.')[0])+".xml")
print("\n")
do_something_with_try_statements(root)
# do_something_with_throw_statements(root)
