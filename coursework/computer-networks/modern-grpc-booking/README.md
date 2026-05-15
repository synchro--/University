# modern-grpc-booking

Minimal **gRPC** reimplementation of the `simulazione_2013` Java RMI **booking** exercise (`Prenotazione`: view + delete). No file transfer, no RMI — only an in-memory server and a stdin client.

## Prerequisites

- JDK 17+
- Network access on first build (Gradle downloads dependencies)

## Build

```bash
cd coursework/computer-networks/modern-grpc-booking
./gradlew build
```

On Windows: `gradlew.bat build`. If the wrapper is missing, install Gradle and run `gradle wrapper`, or use a local Gradle install.

## Run

**Terminal 1 — server** (port `50051`):

```bash
./gradlew runServer
```

**Terminal 2 — client**:

```bash
./gradlew runClient
```

Optional host/port: `./gradlew runClient --args="localhost 50051"`.

### Client commands (same as RMI original)

| Key | Action |
|-----|--------|
| `V` | List bookings matching tipo (`piazzola deluxe`, `piazzola`, `mezza piazzola`) and minimum people |
| `E` | Delete booking by id (tries to remove image file, then clears the slot) |

## Layout

| Path | Role |
|------|------|
| `proto/booking.proto` | Service + messages (Protobuf) |
| `booking.BookingServer` | gRPC server |
| `booking.BookingClient` | Interactive CLI |
| `booking.BookingServiceImpl` | In-memory store (5 slots, one test row) |

## vs Java RMI (2013)

| RMI | This project |
|-----|----------------|
| `Naming.lookup("//host:1099/server_RMI")` | `ManagedChannel` + generated stub |
| `RMI_interfaceFile extends Remote` | `BookingService` in `.proto` |
| `Prenotazione implements Serializable` | `Booking` protobuf message |
| JDK registry on 1099 | gRPC on 50051 |

Modern services use gRPC, REST, or GraphQL over HTTP/2 instead of the JDK RMI registry.
