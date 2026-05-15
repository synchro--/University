package booking;

import booking.grpc.Booking;
import booking.grpc.BookingServiceGrpc;
import booking.grpc.DeleteBookingRequest;
import booking.grpc.ListBookingsRequest;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

/**
 * Interactive CLI — same commands as legacy {@code RMI_Client} (V / E).
 */
public final class BookingClient {
  private final ManagedChannel channel;
  private final BookingServiceGrpc.BookingServiceBlockingStub stub;

  public BookingClient(String host, int port) {
    channel = ManagedChannelBuilder.forAddress(host, port)
        .usePlaintext()
        .build();
    stub = BookingServiceGrpc.newBlockingStub(channel);
  }

  public void shutdown() throws InterruptedException {
    channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
  }

  public void runInteractive() {
    Scanner in = new Scanner(new BufferedReader(new InputStreamReader(System.in)));
    System.out.println("gRPC booking client connected to " + BookingServer.PORT);
    System.out.println("\nV=visualizza prenotazione\nE=elimina prenotazione\nEOF per terminare :");

    while (in.hasNextLine()) {
      String command = in.nextLine().trim();
      if (command.equalsIgnoreCase("E")) {
        System.out.println("Inserisci id prenotazione");
        String id = in.nextLine();
        int esito = stub.deleteBooking(DeleteBookingRequest.newBuilder().setId(id).build()).getResult();
        if (esito == 0) {
          System.out.println("eliminazione effettuata correttamente");
        } else if (esito == 1) {
          System.out.println("errore nell'eliminazione del file_IMg");
        } else {
          System.out.println("Prenotazione non esistente");
        }
      } else if (command.equalsIgnoreCase("V")) {
        String tipo;
        do {
          System.out.println("Inserisci tipo");
          tipo = in.nextLine();
        } while (!tipo.equals("piazzola deluxe") && !tipo.equals("piazzola") && !tipo.equals("mezza piazzola"));

        System.out.println("Inserisci soglia persone");
        int numPersone;
        try {
          numPersone = Integer.parseInt(in.nextLine());
        } catch (NumberFormatException e) {
          System.out.println("La soglia deve essere intera!");
          continue;
        }

        for (Booking p : stub.listBookings(
            ListBookingsRequest.newBuilder().setBookingType(tipo).setMinPeople(numPersone).build())
            .getBookingsList()) {
          System.out.println(format(p));
        }
      }

      System.out.println("\nV=visualizza prenotazione\nE=elimina prenotazione\nEOF per terminare :");
    }
  }

  private static String format(Booking p) {
    return p.getId() + "  " + p.getNumPeople() + "  " + p.getBookingType() + "  "
        + p.getVehicle() + "  " + p.getPlate() + "  " + p.getImageFile();
  }

  public static void main(String[] args) throws InterruptedException {
    String host = args.length > 0 ? args[0] : "localhost";
    int port = args.length > 1 ? Integer.parseInt(args[1]) : BookingServer.PORT;

    BookingClient client = new BookingClient(host, port);
    try {
      client.runInteractive();
    } finally {
      client.shutdown();
    }
  }
}
