from fastmcp import FastMcp

mcp = FastMcp(name = 'Demo try')

@mcp.tool()
def add(a: float, b: float) -> float:
    """add two number together"""
    return a + b

if __name__ == '__main__':
    mcp.run(transport = 'stdio')
 
 