package com.epicmonstrosity.brewference.tokenizer.decoder;

import com.epicmonstrosity.brewference.generation.TokenConsumer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

public class StreamingUnicodeDecoder {
    private final ByteArrayOutputStream pending = new ByteArrayOutputStream();
    private final TokenConsumer consumer;

    public StreamingUnicodeDecoder(final TokenConsumer consumer) {
        this.consumer = consumer;
    }

    public void append(final byte[] tokenBytes) {
        append(0, 0, tokenBytes);
    }

    public void append(final int position, final int tokenId, final byte[] tokenBytes) {
        pending.write(tokenBytes, 0, tokenBytes.length);
        final byte[] buf = pending.toByteArray();
        final int complete = lengthOfCompleteUtf8(buf);
        if (complete > 0) {
            consumer.onGeneratedToken(position, tokenId, new String(buf, 0, complete, StandardCharsets.UTF_8));
            pending.reset();
            pending.write(buf, complete, buf.length - complete);
        }
    }

    public void flush(final int position, final int tokenId) {
        final byte[] buf = pending.toByteArray();
        if (buf.length > 0) {
            consumer.onGeneratedToken(position, tokenId, new String(buf, StandardCharsets.UTF_8));
            pending.reset();
        }
    }

    private static int lengthOfCompleteUtf8(final byte[] buf) {
        // Walk back from the end past any trailing incomplete sequence
        int i = buf.length;
        int back = 0;
        while (i > 0 && back < 4) {
            final int b = buf[--i] & 0xFF;
            back++;
            if ((b & 0xC0) != 0x80) {
                final int need = determineUtf8CharacterSize(b);
                return (back >= need) ? buf.length : i;
            }
        }
        return buf.length;
    }

    private static int determineUtf8CharacterSize(final int byteValue) {
        if ((byteValue & 0x80) == 0x00)
            return 1;
        if ((byteValue & 0xE0) == 0xC0)
            return 2;
        if ((byteValue & 0xF0) == 0xE0)
            return 3;
        if ((byteValue & 0xF8) == 0xF0)
            return 4;
        return 1;
    }
}