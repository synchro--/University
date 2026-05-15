# Computer Networks coursework

University exercises on sockets and distributed Java (circa 2013). Kept for reference; the Java RMI parts are obsolete in production but still useful to understand how remote calls evolved.

## `simulazione_2013/` — sockets + RMI booking

Two halves of the same exam-style simulation:

| Part | Tech | What it does |
|------|------|----------------|
| **C/** | BSD sockets (`select`, TCP/UDP) | Simple client/server file transfer. Educational only (uses deprecated APIs like `gets()`). |
| **Java/** | Java RMI | Remote **booking** service: view and delete `Prenotazione` records. |

### RMI booking flow (Java)

Classic Java RMI pattern from the course:

1. **Registry** — `RMI_Server` binds to `//localhost:1099/server_RMI` via `Naming.rebind`.
2. **Stub lookup** — `RMI_Client` does `Naming.lookup(complete)` and casts to `RMI_interfaceFile`.
3. **Remote interface** — `RMI_interfaceFile extends Remote` declares `elimina_prenotazione` and `visualizza_prenotazione`.
4. **Serializable DTO** — `Prenotazione implements Serializable` is passed by value over the wire.
5. **Interactive client** — stdin loop: `V` = list by tipo + min people, `E` = delete by id.

The server keeps a fixed array of in-memory bookings and test data in `init()`.

### Modern equivalent

See **`modern-grpc-booking/`** — same booking view/delete semantics with **gRPC + Protocol Buffers** (no RMI). That is the stack used today (gRPC, REST, GraphQL) instead of Java’s built-in RMI registry.

## `estensione_6/` — RMI metadata + socket file transfer

Hybrid design common in older courses:

1. **RMI phase** — client gets directory list, file names, sizes, and a **socket endpoint** (`RemoteInfo`, `FileInfo`).
2. **Socket phase** — actual bytes move over **TCP stream sockets** (`ActiveThread`, `PassiveConThread`).

RMI was often used only for control plane; bulk data went over raw sockets.

## Build / run (legacy Java RMI)

From `simulazione_2013/Java/` (requires a JDK with `rmic` if you still compile stubs the old way, or run on a JDK that supports classic RMI):

```bash
javac *.java
# terminal 1
java RMI_Server
# terminal 2
java RMI_Client
```

Client prompts: `V` / `E`, then booking fields as in the original assignment.
