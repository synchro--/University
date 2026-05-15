package booking;

import io.grpc.Server;
import io.grpc.ServerBuilder;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public final class BookingServer {
    public static final int PORT = 50051;

    private Server server;

    private void start() throws IOException {
        server = ServerBuilder.forPort(PORT)
                .addService(new BookingServiceImpl())
                .build()
                .start();
        System.out.println("gRPC booking server listening on port " + PORT);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                BookingServer.this.stop();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }));
    }

    private void stop() throws InterruptedException {
        if (server != null) {
            server.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    private void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        BookingServer app = new BookingServer();
        app.start();
        app.blockUntilShutdown();
    }
}
