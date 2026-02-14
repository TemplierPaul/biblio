# Databases & SQL - Interview Q&A

Essential database concepts for ML roles, covering SQL mastery, NoSQL paradigms, and query optimization.

---

## Table of Contents

- [[#Part 1: SQL Fundamentals]]
  - [[#Order of Execution in SQL?]]
  - [[#Explain the different types of Joins (Inner, Left, Right, Full, Cross)]]
  - [[#What are Window Functions and why are they useful?]]
  - [[#Explain Common Table Expressions (CTEs)]]
  - [[#Difference between WHERE and HAVING?]]
- [[#Part 2: Database Design]]
  - [[#What is Normalization? (1NF, 2NF, 3NF)]]
  - [[#OLTP vs OLAP - Key Differences?]]
  - [[#Star Schema vs Snowflake Schema?]]
- [[#Part 3: NoSQL Databases]]
  - [[#Key-Value Stores (Redis) - Use Cases?]]
  - [[#Document Stores (MongoDB) - Use Cases?]]
  - [[#Wide-Column Stores (Cassandra) - Use Cases?]]
  - [[#Graph Databases (Neo4j) - Use Cases?]]
- [[#Part 4: Optimization & Internals]]
  - [[#How do Indexes work? (B-Tree vs Hash)]]
  - [[#What is an Explain Plan?]]
  - [[#Columnar vs Row-oriented Storage?]]
  - [[#Database Sharding vs Partitioning?]]

---

## Part 1: SQL Fundamentals

### Order of Execution in SQL?

SQL queries are executed in a specific order, not the order they are written:
1.  **FROM / JOIN**: Determine the data source.
2.  **WHERE**: Filter rows.
3.  **GROUP BY**: Aggregate rows.
4.  **HAVING**: Filter groups.
5.  **SELECT**: Select columns.
6.  **DISTINCT**: Remove duplicates.
7.  **ORDER BY**: Sort results.
8.  **LIMIT / OFFSET**: Restrict results.

### Explain the different types of Joins (Inner, Left, Right, Full, Cross)

*   **INNER JOIN**: Returns rows when there is a match in *both* tables.
*   **LEFT JOIN**: Returns all rows from the *left* table, and the matched rows from the right table. (Null if no match).
*   **RIGHT JOIN**: Returns all rows from the *right* table, and the matched rows from the left table.
*   **FULL OUTER JOIN**: Returns all rows when there is a match in *either* left or right table.
*   **CROSS JOIN**: Returns the Cartesian product of the two tables (Row count = Table1_rows * Table2_rows).

### What are Window Functions and why are they useful?

Window functions perform calculations across a set of table rows that are somehow related to the current row. Unlike aggregate functions, they do **not** cause rows to become grouped into a single output row.

**Syntax**: `FUNCTION() OVER (PARTITION BY col1 ORDER BY col2)`

**Common Functions**:
*   `ROW_NUMBER()`: Unique cumulative number (1, 2, 3...)
*   `RANK()`: Ranking with gaps (1, 1, 3...)
*   `DENSE_RANK()`: Ranking without gaps (1, 1, 2...)
*   `LEAD() / LAG()`: Access next/previous row values.
*   `MOVING AVERAGE`: `AVG(val) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)`

### Explain Common Table Expressions (CTEs)

A temporary result set that you can reference within a `SELECT`, `INSERT`, `UPDATE`, or `DELETE` statement.
**Syntax**: `WITH CteName AS (SELECT ...) SELECT * FROM CteName`
**Benefits**: Readability, modularity, recursion (e.g., traversing hierarchical data like org charts).

### Difference between WHERE and HAVING?

*   **WHERE**: Filters rows *before* aggregation. Cannot use aggregate functions (e.g., `SUM`, `COUNT`).
*   **HAVING**: Filters groups *after* aggregation. Used with `GROUP BY`.

---

## Part 2: Database Design

### What is Normalization? (1NF, 2NF, 3NF)

Process of organizing data to reduce redundancy and improve data integrity.
*   **1NF**: Atomic values (no lists in columns), unique rows.
*   **2NF**: 1NF + No partial dependencies (all non-key columns depend on the *entire* primary key).
*   **3NF**: 2NF + No transitive dependencies (non-key columns depend *only* on the primary key, not other non-key columns).

**Trade-off**: High normalization = more joins (slower reads), better writes. Denormalization = faster reads, redundant data (risk of inconsistency).

### OLTP vs OLAP - Key Differences?

| Feature | OLTP (Online Transaction Processing) | OLAP (Online Analytical Processing) |
| :--- | :--- | :--- |
| **Focus** | Daily transactions (CRUD) | Analysis, Reporting, Intelligence |
| **Queries** | Simple, specific (lookup by ID) | Complex, aggregations over millions of rows |
| **Design** | Normalized (3NF) | Denormalized (Star/Snowflake) |
| **Latency** | Low (ms) | High (seconds/minutes) |
| **Example** | PostgreSQL, MySQL (User DB) | Snowflake, BigQuery, Redshift (Data Warehouse) |

### Star Schema vs Snowflake Schema?

*   **Star Schema**: Central "Fact Table" (metrics) connected to "Dimension Tables" (attributes). Dimensions are denormalized. Simpler joins, faster queries.
*   **Snowflake Schema**: Dimensions are normalized into multiple related tables. Saves storage, complicates queries (more joins).

---

## Part 3: NoSQL Databases

### Key-Value Stores (Redis) - Use Cases?

*   **Data Model**: Dictionary (Hash Map).
*   **Use Cases**: Caching (User sessions), Leaderboards, Real-time counting, Message Broker (Pub/Sub).
*   **Pros**: Ultra-low latency (in-memory).
*   **Cons**: Limited query capability (lookup by key only).

### Document Stores (MongoDB) - Use Cases?

*   **Data Model**: JSON-like documents (flexible schema).
*   **Use Cases**: Content management (CMS), User profiles, Catalogs, Prototyping (rapid schema evolution).
*   **Pros**: Flexible schema, maps naturally to code objects.
*   **Cons**: Consistency challenges, complex joins are hard.

### Wide-Column Stores (Cassandra) - Use Cases?

*   **Data Model**: Distributed multi-dimensional map.
*   **Use Cases**: Time-series data (IoT sensors), Activity logs, Message history (Discord/Slack).
*   **Pros**: Massive write throughput (append-only), linear scalability.
*   **Cons**: Eventual consistency, rigorous schema design required upfront (query-driven design).

### Graph Databases (Neo4j) - Use Cases?

*   **Data Model**: Nodes and Edges.
*   **Use Cases**: Social networks, Recommendation engines, Fraud detection (circular transactions), Knowledge graphs.
*   **Pros**: Efficient traversal of relationships (O(1) hop vs SQL Join O(log N)).

---

## Part 4: Optimization & Internals

### How do Indexes work? (B-Tree vs Hash)

*   **B-Tree**: Balanced tree structure. Sorted data.
    *   Supports: Equality (`=`), Range (`<`, `>`, `BETWEEN`), Sorting (`ORDER BY`).
    *   Complexity: O(log N) search/insert/delete.
    *   Default for most DBs.
*   **Hash Index**: Hash table.
    *   Supports: Equality (`=`) only.
    *   Complexity: O(1) search.
    *   Cannot handle range queries or sorting.

### What is an Explain Plan?

A breakdown of how the database engine intends to execute a query.
**Look for**:
*   **Full Table Scans**: Reading every row (Bad for large tables).
*   **Index Scans/Seeks**: Using index (Good).
*   **Sort operations**: Expensive (try to use index for sorting).
*   **Join types**: Nested Loop vs Hash Join vs Merge Join.

### Columnar vs Row-oriented Storage?

*   **Row-oriented** (Postgres/MySQL): Stores data row by row.
    *   Good for: Fetching all info for one entity (User profile), Transactions (INSERT/UPDATE).
*   **Columnar** (Parquet/BigQuery): Stores data column by column.
    *   Good for: Analytics (SUM(revenue)), Compression (similar data types together).
    *   Bad for: Frequent updates, fetching single rows.

### Database Sharding vs Partitioning?

*   **Partitioning**: Splitting a table into smaller chunks *within the same database instance* (e.g., by Date).
*   **Sharding**: Horizontal scaling. Splitting data across *multiple database servers* (Instances).
    *   **Challenge**: Cross-shard joins are expensive/impossible. Rebalancing data is hard.
